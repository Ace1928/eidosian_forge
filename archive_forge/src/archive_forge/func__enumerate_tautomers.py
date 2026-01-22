from warnings import warn
import copy
import logging
from rdkit import Chem
from rdkit.Chem.rdchem import BondDir, BondStereo, BondType
from .utils import memoized_property, pairwise
@memoized_property
def _enumerate_tautomers(self):
    return TautomerEnumerator(self.transforms, self.max_tautomers)