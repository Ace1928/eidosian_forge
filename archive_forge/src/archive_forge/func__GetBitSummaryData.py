import copy
import numpy
from rdkit.Chem.Pharm2D import Utils
from rdkit.DataStructs import (IntSparseIntVect, LongSparseIntVect,
def _GetBitSummaryData(self, bitIdx):
    nPts, combo, scaffold = self.GetBitInfo(bitIdx)
    fams = self.GetFeatFamilies()
    labels = [fams[x] for x in combo]
    dMat = numpy.zeros((nPts, nPts), dtype=numpy.int64)
    dVect = Utils.nPointDistDict[nPts]
    for idx in range(len(dVect)):
        i, j = dVect[idx]
        dMat[i, j] = scaffold[idx]
        dMat[j, i] = scaffold[idx]
    return (nPts, combo, scaffold, labels, dMat)