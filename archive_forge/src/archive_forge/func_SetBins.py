import copy
import numpy
from rdkit.Chem.Pharm2D import Utils
from rdkit.DataStructs import (IntSparseIntVect, LongSparseIntVect,
def SetBins(self, bins):
    """ bins should be a list of 2-tuples """
    self._bins = copy.copy(bins)
    self.Init()