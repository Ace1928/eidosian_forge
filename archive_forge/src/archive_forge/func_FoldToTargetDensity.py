from rdkit import rdBase
from rdkit.DataStructs import cDataStructs
from rdkit.DataStructs.cDataStructs import *
def FoldToTargetDensity(fp, density=0.3, minLength=64):
    while fp.GetNumOnBits() / len(fp) > density and len(fp) // 2 > minLength:
        fp = FoldFingerprint(fp, 2)
    return fp