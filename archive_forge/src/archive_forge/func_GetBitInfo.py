import copy
import numpy
from rdkit.Chem.Pharm2D import Utils
from rdkit.DataStructs import (IntSparseIntVect, LongSparseIntVect,
def GetBitInfo(self, idx):
    """ returns information about the given bit

         **Arguments**

           - idx: the bit index to be considered

         **Returns**

           a 3-tuple:

             1) the number of points in the pharmacophore

             2) the proto-pharmacophore (tuple of pattern indices)

             3) the scaffold (tuple of distance indices)

        """
    if idx >= self._sigSize:
        raise IndexError(f'bad index ({idx}) queried. {self._sigSize} is the max')
    nPts = self.minPointCount
    while nPts < self.maxPointCount and self._starts[nPts + 1] <= idx:
        nPts += 1
    offsetFromStart = idx - self._starts[nPts]
    if _verbose:
        print(f'\t {nPts} Points, {offsetFromStart} offset')
    nDists = len(Utils.nPointDistDict[nPts])
    scaffolds = self._scaffolds[nDists]
    nScaffolds = len(scaffolds)
    protoIdx = offsetFromStart // nScaffolds
    indexCombos = Utils.GetIndexCombinations(self._nFeats, nPts)
    combo = tuple(indexCombos[protoIdx])
    if _verbose:
        print(f'\t combo: {str(combo)}')
    scaffoldIdx = offsetFromStart % nScaffolds
    scaffold = scaffolds[scaffoldIdx]
    if _verbose:
        print(f'\t scaffold: {str(scaffold)}')
    return (nPts, combo, scaffold)