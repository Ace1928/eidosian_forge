import os
import re
import weakref
from rdkit import Chem, RDConfig
def CreateMolFingerprint(mol, hierarchy):
    totL = 0
    for entry in hierarchy:
        totL += len(entry)
    res = [0] * totL
    idx = 0
    for entry in hierarchy:
        idx = _SetNodeBits(mol, entry, res, idx)
    return res