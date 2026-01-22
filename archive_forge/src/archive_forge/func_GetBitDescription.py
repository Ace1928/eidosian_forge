import copy
import numpy
from rdkit.Chem.Pharm2D import Utils
from rdkit.DataStructs import (IntSparseIntVect, LongSparseIntVect,
def GetBitDescription(self, bitIdx):
    """  returns a text description of the bit

        **Arguments**

          - bitIdx: an integer bit index

        **Returns**

          a string

        """
    _, _, _, labels, dMat = self._GetBitSummaryData(bitIdx)
    res = ' '.join(labels) + ' '
    for row in dMat:
        res += '|' + ' '.join([str(x) for x in row])
    res += '|'
    return res