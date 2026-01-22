import itertools
from collections import Counter, defaultdict
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdqueries
from . import utils
def _calcReactivityMolecule(self, mol):
    reactivityAtoms = [self._calcReactivityAtom(a) for a in mol.GetAtoms()]
    return reactivityAtoms