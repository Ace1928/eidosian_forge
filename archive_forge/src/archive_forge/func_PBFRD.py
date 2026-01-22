import numpy as np
from numpy import linalg
from rdkit import Chem
from rdkit.Chem import AllChem
def PBFRD(mol, confId=-1):
    conf = mol.GetConformer(confId)
    if not conf.Is3D():
        return 0
    pts = np.array([list(conf.GetAtomPosition(x)) for x in range(mol.GetNumAtoms())])
    plane = GetBestFitPlane(pts)
    denom = np.dot(plane[:3], plane[:3])
    denom = denom ** 0.5
    res = 0.0
    for pt in pts:
        res += np.abs(pt.dot(plane[:3]) + plane[3])
    res /= denom
    res /= len(pts)
    return res