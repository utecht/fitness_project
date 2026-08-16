#!/usr/bin/env python3
"""
hr_by_mile.py — per-mile heart-rate distribution viewer for a running .fit file.

Usage:
  python3 hr_by_mile.py RUN.fit [-o out.html] [--cap 145] [--watch 150] [--title "Sat 8/15 long run"]

Running-only samples (speed >= 1.79 m/s, i.e. faster than ~15:00/mi) so stops,
lights, and walk breaks don't smear the distributions. Partial final mile is
included if >= 0.3 mi and marked with *.
"""
import argparse, json, sys, os
from datetime import timedelta
import numpy as np
from fitparse import FitFile

TEMPLATE = '<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">\n<title>HR by mile · __TITLE__</title>\n<style>\n:root{--bg:#0f1115;--panel:#161a22;--panel2:#1c2230;--border:#262d3d;--text:#e6e8ee;--muted:#8a93a6;--accent:#7cc4ff;--z1:#3a4a6b;--z2:#1f8a5a;--z3:#b3741b;--z4:#b23b3b;--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}\n@media(prefers-color-scheme:light){:root{--bg:#f7f8fa;--panel:#fff;--panel2:#f0f3f8;--border:#e2e6ee;--text:#1a1d23;--muted:#5f6774;--accent:#1566c0}}\n*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}\n.wrap{max-width:1040px;margin:0 auto;padding:20px}\nh1{font-size:18px;font-weight:600;margin:0}.sub{color:var(--muted);font-size:13px;margin-top:2px}\n.controls{display:flex;gap:18px;flex-wrap:wrap;align-items:center;margin:16px 0;font-size:13px;color:var(--muted)}\n.controls label{display:flex;align-items:center;gap:6px}\ninput[type=number]{width:62px;background:var(--panel2);border:1px solid var(--border);color:var(--text);border-radius:6px;padding:4px 6px;font:inherit;font-family:var(--mono)}\ninput[type=checkbox]{accent-color:var(--accent)}\n.card{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:14px 16px 10px}\nsvg{width:100%;height:auto;display:block}\n.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin-top:10px}\n.legend span::before{content:"";display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px;background:var(--c)}\n.note{color:var(--muted);font-size:12px;margin-top:12px;line-height:1.5}\n</style></head><body><div class="wrap">\n<h1>Heart rate distribution by mile</h1>\n<div class="sub">__SUB__</div>\n<div class="controls">\n<label>Easy cap <input type="number" id="cap" value="145"></label>\n<label>Watch Z2 ceiling <input type="number" id="watch" value="150"></label>\n<label><input type="checkbox" id="showDensity" checked> density</label>\n<label><input type="checkbox" id="showBox" checked> quartiles</label>\n</div>\n<div class="card"><svg id="chart" viewBox="0 0 1000 __VBH__" role="img" aria-label="Heart rate distribution per mile"></svg>\n<div class="legend"><span style="--c:var(--z1)">Z1 &lt;125</span><span style="--c:var(--z2)">Z2 125–cap</span><span style="--c:var(--z3)">Z3 cap+1–155</span><span style="--c:var(--z4)">Z4 156+</span><span style="--c:var(--accent)">median · box p25–p75 · whisker p10–p90</span><span style="--c:var(--z3)">▲ climb</span><span style="--c:var(--accent)">▼ descent</span> · pace colored fast→slow (green→amber→red, OKLCH constant lightness)</div></div>\n<div class="note" id="note"></div>\n</div>\n<script>\nconst M=__DATA__;\nM.forEach(m=>{const [a,b]=m.pace.split(\':\').map(Number);m.paceSec=a*60+b});\nconst pMin=Math.min(...M.map(m=>m.paceSec)),pMax=Math.max(...M.map(m=>m.paceSec)),gMax=Math.max(...M.map(m=>Math.max(m.gain,m.loss)));\nfunction paceColor(sec){const t=(pMax-sec)/(pMax-pMin);const h=25+t*120;return `oklch(0.74 0.16 ${h.toFixed(1)})`}\nconst svg=document.getElementById(\'chart\');\nconst X0=120,X1=165,W=1000,H=620,padL=130,padR=150,padTop=34,rowH=44;\nconst x=v=>padL+(v-X0)/(X1-X0)*(W-padL-padR);\nfunction render(){\n const cap=+document.getElementById(\'cap\').value,watch=+document.getElementById(\'watch\').value;\n const dens=document.getElementById(\'showDensity\').checked,box=document.getElementById(\'showBox\').checked;\n let s=\'\';\n // zone bands\n const bands=[[X0,125,\'var(--z1)\'],[125,cap,\'var(--z2)\'],[cap,155,\'var(--z3)\'],[155,X1,\'var(--z4)\']];\n bands.forEach(([a,b,c])=>{s+=`<rect x="${x(a)}" y="${padTop}" width="${x(b)-x(a)}" height="${rowH*M.length}" fill="${c}" opacity="0.13"/>`});\n // grid + axis\n for(let v=X0;v<=X1;v+=5){s+=`<line x1="${x(v)}" y1="${padTop}" x2="${x(v)}" y2="${padTop+rowH*M.length}" stroke="var(--border)" stroke-width="1"/>`;\n  s+=`<text x="${x(v)}" y="${padTop-10}" font-size="11" fill="var(--muted)" text-anchor="middle" font-family="var(--mono)">${v}</text>`}\n // cap & watch lines\n s+=`<line x1="${x(cap)}" y1="${padTop-4}" x2="${x(cap)}" y2="${padTop+rowH*M.length+4}" stroke="var(--z3)" stroke-width="2"/>`;\n s+=`<text x="${x(cap)+4}" y="${padTop+rowH*M.length+16}" font-size="11" fill="var(--z3)" font-family="var(--mono)">cap ${cap}</text>`;\n s+=`<line x1="${x(watch)}" y1="${padTop-4}" x2="${x(watch)}" y2="${padTop+rowH*M.length+4}" stroke="var(--muted)" stroke-width="1.5" stroke-dasharray="4 4"/>`;\n s+=`<text x="${x(watch)+4}" y="${padTop+rowH*M.length+30}" font-size="11" fill="var(--muted)" font-family="var(--mono)">watch Z2 padTop ${watch}</text>`;\n // header labels\n s+=`<text x="12" y="${padTop-10}" font-size="11" fill="var(--muted)">MILE · PACE</text>`;\n s+=`<text x="${W-padR+8}" y="${padTop-10}" font-size="11" fill="var(--muted)">≤CAP · CLIMB ft</text>`;\n M.forEach((m,i)=>{\n  const y=padTop+i*rowH,cy=y+rowH/2;\n  const n=m.n; let under=0;const keys=Object.keys(m.hist).map(Number);\n  keys.forEach(k=>{if(k<=cap)under+=m.hist[k]});\n  const pct=Math.round(under/n*100);\n  // density polygon (2-bpm smoothing)\n  if(dens){let maxc=0;const arr=[];for(let v=X0;v<=X1;v++){const c=((m.hist[v-1]||0)+(m.hist[v]||0)*2+(m.hist[v+1]||0))/4;arr.push([v,c]);if(c>maxc)maxc=c}\n   const sc=(rowH*0.42)/maxc;let p=`M${x(X0)},${cy}`;arr.forEach(([v,c])=>p+=` L${x(v)},${cy-c*sc}`);p+=` L${x(X1)},${cy}`;\n   let q=`M${x(X0)},${cy}`;arr.forEach(([v,c])=>q+=` L${x(v)},${cy+c*sc}`);q+=` L${x(X1)},${cy}`;\n   s+=`<path d="${p}" fill="var(--accent)" opacity="0.28"/><path d="${q}" fill="var(--accent)" opacity="0.28"/>`}\n  if(box){s+=`<line x1="${x(m.p10)}" y1="${cy}" x2="${x(m.p90)}" y2="${cy}" stroke="var(--text)" stroke-width="1.5" opacity="0.7"/>`;\n   s+=`<rect x="${x(m.p25)}" y="${cy-7}" width="${x(m.p75)-x(m.p25)}" height="14" fill="var(--panel)" stroke="var(--text)" stroke-width="1.5" rx="2"/>`;\n   s+=`<line x1="${x(m.p50)}" y1="${cy-9}" x2="${x(m.p50)}" y2="${cy+9}" stroke="var(--accent)" stroke-width="3"/>`}\n  s+=`<text x="12" y="${cy+4}" font-size="13" fill="var(--text)" font-weight="600">Mi ${m.mile}</text>`;\n  s+=`<text x="58" y="${cy+4}" font-size="13" font-weight="600" fill="${paceColor(m.paceSec)}" font-family="var(--mono)">${m.pace}</text>`;\n  const pc=pct>=85?\'var(--z2)\':pct>=60?\'var(--z3)\':\'var(--z4)\';\n  s+=`<text x="${W-padR+8}" y="${cy+4}" font-size="12" font-family="var(--mono)" fill="${pc}">${pct}%</text>`;\n  const bx=W-padR+48,bw=50,gw=m.gain/gMax*bw,lw=m.loss/gMax*bw;\n  s+=`<rect x="${bx}" y="${cy-9}" width="${gw}" height="7" fill="var(--z3)" rx="1.5"/><text x="${bx+gw+4}" y="${cy-2}" font-size="10" font-family="var(--mono)" fill="var(--z3)">▲${m.gain}</text>`;\n  s+=`<rect x="${bx}" y="${cy+2}" width="${lw}" height="7" fill="var(--accent)" opacity="0.7" rx="1.5"/><text x="${bx+lw+4}" y="${cy+9}" font-size="10" font-family="var(--mono)" fill="var(--accent)">▼${m.loss}</text>`;\n  s+=`<line x1="${padL}" y1="${y+rowH}" x2="${W-padR}" y2="${y+rowH}" stroke="var(--border)"/>`;\n });\n svg.innerHTML=s;\n // note\n const overallUnder=M.reduce((a,m)=>a+Object.entries(m.hist).reduce((b,[k,c])=>b+(+k<=cap?c:0),0),0),N=M.reduce((a,m)=>a+m.n,0);\n const overallWatch=M.reduce((a,m)=>a+Object.entries(m.hist).reduce((b,[k,c])=>b+(+k<=watch?c:0),0),0);\n document.getElementById(\'note\').innerHTML=`Whole run: <b>${Math.round(overallUnder/N*100)}%</b> at or under ${cap} · <b>${Math.round(overallWatch/N*100)}%</b> at or under the watch\'s ${watch}. The gap between those two numbers is the disagreement you\'re feeling. Zone bands use running zones (Z2 125–cap). Set the watch ceiling to whatever your Watch app shows for Z2 to see how much of "high Z2" lands in Z3 here.`;\n}\n[\'cap\',\'watch\',\'showDensity\',\'showBox\'].forEach(id=>document.getElementById(id).addEventListener(\'input\',render));\nrender();\n</script></body></html>'

def load(path):
    f = FitFile(path)
    recs = [{fl.name: fl.value for fl in m} for m in f.get_messages('record')]
    g = lambda k: np.array([r.get(k, np.nan) for r in recs], dtype=float)
    ts = [r['timestamp'] for r in recs]
    return dict(hr=g('heart_rate'), spd=g('enhanced_speed'), dist=g('distance'),
                alt=g('enhanced_altitude'), start=ts[0], end=ts[-1])

def per_mile(d, run_thresh=1.79):
    MI = 1609.34
    run = d['spd'] >= run_thresh
    total_mi = np.nanmax(d['dist']) / MI
    n_full = int(total_mi)
    miles = list(range(n_full)) + ([n_full] if total_mi - n_full >= 0.3 else [])
    out = []
    for m in miles:
        sel = (d['dist'] >= m*MI) & (d['dist'] < (m+1)*MI)
        r = sel & run
        h = d['hr'][r]; h = h[~np.isnan(h)]
        if len(h) < 30: continue
        pace = 26.8224 / np.nanmean(d['spd'][r])
        a = d['alt'][sel]; a = a[~np.isnan(a)]
        gain = loss = 0
        if len(a) > 12:
            sm = np.convolve(a, np.ones(10)/10, 'valid'); dz = np.diff(sm)*3.281
            gain, loss = int(np.sum(np.clip(dz,0,None))), int(np.sum(np.clip(-dz,0,None)))
        hist = {}
        for v in h.astype(int): hist[int(v)] = hist.get(int(v),0)+1
        out.append(dict(mile=m+1, label=f"Mi {m+1}" + ("*" if m == n_full else ""),
                        pace=f"{int(pace)}:{int(pace%1*60):02d}", avg=round(float(h.mean()),1),
                        max=int(h.max()), min=int(h.min()),
                        p10=int(np.percentile(h,10)), p25=int(np.percentile(h,25)), p50=int(np.percentile(h,50)),
                        p75=int(np.percentile(h,75)), p90=int(np.percentile(h,90)),
                        gain=gain, loss=loss, n=int(len(h)), hist=hist))
    return out, total_mi

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('fit'); ap.add_argument('-o','--out')
    ap.add_argument('--cap', type=int, default=145); ap.add_argument('--watch', type=int, default=150)
    ap.add_argument('--title')
    a = ap.parse_args()
    d = load(a.fit)
    miles, total = per_mile(d)
    if not miles: sys.exit("no running samples found")
    local = d['start'] - timedelta(hours=7)  # PDT; adjust if needed
    title = a.title or local.strftime('%a %-m/%-d run')
    sub = f"{local.strftime('%a %b %-d, %-I:%M %p')} · {total:.2f} mi · running-only samples (stops removed) · 1 sample/sec"
    html = (TEMPLATE.replace('__DATA__', json.dumps(miles)).replace('__TITLE__', title)
            .replace('__SUB__', sub).replace('__VBH__', str(34+44*len(miles)+50))
            .replace('value="145"', f'value="{a.cap}"').replace('value="150"', f'value="{a.watch}"')
            .replace('`Mi ${m.mile}`', '`${m.label}`'))
    out = a.out or os.path.splitext(os.path.basename(a.fit))[0][:10] + '-hr-by-mile.html'
    open(out,'w').write(html); print(out)

if __name__ == '__main__': main()
