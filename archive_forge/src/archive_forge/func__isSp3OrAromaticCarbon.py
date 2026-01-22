import itertools
from collections import Counter, defaultdict
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdqueries
from . import utils
def _isSp3OrAromaticCarbon(self, a):
    if a.GetAtomicNum() != 6:
        return False
    if a.GetIsAromatic():
        return True
    for b in a.GetBonds():
        if b.GetBondTypeAsDouble() > 1.5:
            return False
    return True