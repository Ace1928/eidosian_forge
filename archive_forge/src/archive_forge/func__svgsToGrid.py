import re
from collections import Counter, defaultdict, namedtuple
import numpy as np
import seaborn as sns
from numpy import linalg
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
def _svgsToGrid(svgs, labels, svgsPerRow=4, molSize=(250, 150), fontSize=12):
    matcher = re.compile('^(<.*>\\n)(<rect .*</rect>\\n)(.*)</svg>', re.DOTALL)
    hdr = ''
    ftr = '</svg>'
    rect = ''
    nRows = len(svgs) // svgsPerRow
    if len(svgs) % svgsPerRow:
        nRows += 1
    blocks = [''] * (nRows * svgsPerRow)
    labelSizeDist = fontSize * 5
    fullSize = (svgsPerRow * (molSize[0] + molSize[0] / 10.0), nRows * (molSize[1] + labelSizeDist))
    count = 0
    for svg, name in zip(svgs, labels):
        h, r, b = matcher.match(svg).groups()
        if hdr == '':
            hdr = h.replace("width='{}px'".format(molSize[0]), "width='{}px'".format(fullSize[0]))
            hdr = hdr.replace("height='{}px'".format(molSize[1]), "height='{}px'".format(fullSize[1]))
        if rect == '':
            rect = r
        tspanFmt = '<tspan x="{0}" y="{1}">{2}</tspan>'
        names = name.split('|')
        legend = []
        legend.append('<text font-family="sans-serif" font-size="{}px" text-anchor="middle" fill="black">'.format(fontSize))
        legend.append(tspanFmt.format(molSize[0] / 2.0, molSize[1] + fontSize * 2, names[0]))
        if len(names) > 1:
            legend.append(tspanFmt.format(molSize[0] / 2.0, molSize[1] + fontSize * 3.5, names[1]))
        legend.append('</text>')
        legend = '\n'.join(legend)
        blocks[count] = b + legend
        count += 1
    for i, elem in enumerate(blocks):
        row = i // svgsPerRow
        col = i % svgsPerRow
        elem = rect + elem
        blocks[i] = '<g transform="translate(%d,%d)" >%s</g>' % (col * (molSize[0] + molSize[0] / 10.0), row * (molSize[1] + labelSizeDist), elem)
    res = hdr + '\n'.join(blocks) + ftr
    return res