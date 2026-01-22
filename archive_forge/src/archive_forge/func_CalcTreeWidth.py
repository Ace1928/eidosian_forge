import math
from rdkit.sping import pid as piddle
def CalcTreeWidth(tree):
    try:
        tree.totNChildren
    except AttributeError:
        CalcTreeNodeSizes(tree)
    totWidth = tree.totNChildren * (visOpts.circRad + visOpts.horizOffset)
    return totWidth