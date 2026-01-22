import itertools
from collections import Counter, defaultdict
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdqueries
from . import utils
def _atomDetailInvariant(self, mol):
    mol.UpdatePropertyCache(False)
    num_atoms = mol.GetNumAtoms()
    Chem.GetSSSR(mol)
    rinfo = mol.GetRingInfo()
    invariants = [0] * num_atoms
    for i, a in enumerate(mol.GetAtoms()):
        descriptors = []
        descriptors.append(a.GetAtomicNum())
        descriptors.append(a.GetTotalDegree())
        descriptors.append(a.GetTotalNumHs())
        descriptors.append(rinfo.IsAtomInRingOfSize(a.GetIdx(), 6))
        descriptors.append(rinfo.IsAtomInRingOfSize(a.GetIdx(), 5))
        descriptors.append(a.IsInRing())
        descriptors.append(a.GetIsAromatic())
        invariants[i] = hash(tuple(descriptors)) & 4294967295
    return invariants