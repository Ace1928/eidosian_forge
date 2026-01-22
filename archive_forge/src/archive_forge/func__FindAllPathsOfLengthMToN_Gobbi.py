import time
import unittest
import numpy
from scipy.optimize import linear_sum_assignment
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdmolops
from rdkit.Chem.Fingerprints import FingerprintMols
def _FindAllPathsOfLengthMToN_Gobbi(atom, path, minlength, maxlength, visited, paths):
    for bond in atom.GetBonds():
        if bond.GetIdx() not in path:
            bidx = bond.GetIdx()
            path.append(bidx)
            if len(path) >= minlength and len(path) <= maxlength:
                paths.append(tuple(path))
            if len(path) < maxlength:
                a1 = bond.GetBeginAtom()
                a2 = bond.GetEndAtom()
                if a1.GetIdx() == atom.GetIdx():
                    nextatom = a2
                else:
                    nextatom = a1
                nextatomidx = nextatom.GetIdx()
                if nextatomidx not in visited:
                    visited.add(nextatomidx)
                    _FindAllPathsOfLengthMToN_Gobbi(nextatom, path, minlength, maxlength, visited, paths)
                    visited.remove(nextatomidx)
            path.pop()