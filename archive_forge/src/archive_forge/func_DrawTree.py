import math
from rdkit.sping import pid as piddle
def DrawTree(tree, canvas, nRes=2, scaleLeaves=False, allowShrink=True, showPurity=False):
    dims = canvas.size
    loc = (dims[0] / 2, visOpts.vertOffset)
    if scaleLeaves:
        SetNodeScales(tree)
    if allowShrink:
        treeWid = CalcTreeWidth(tree)
        while treeWid > dims[0]:
            visOpts.circRad /= 2
            visOpts.horizOffset /= 2
            treeWid = CalcTreeWidth(tree)
    DrawTreeNode(tree, loc, canvas, nRes, scaleLeaves=scaleLeaves, showPurity=showPurity)