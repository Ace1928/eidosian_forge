import numpy
from rdkit import Geometry
from rdkit.Chem import ChemicalFeatures
from rdkit.RDLogger import logger
def _checkBounds(self, i, j):
    """ raises ValueError on failure """
    nf = len(self._feats)
    if 0 <= i < nf and 0 <= j < nf:
        return True
    raise ValueError('Index out of bound')