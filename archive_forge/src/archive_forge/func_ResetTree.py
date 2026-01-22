import math
from rdkit.sping import pid as piddle
def ResetTree(tree):
    tree._scales = None
    tree.totNChildren = None
    for child in tree.GetChildren():
        ResetTree(child)