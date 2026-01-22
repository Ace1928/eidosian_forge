from fontTools.misc.textTools import bytesjoin, safeEval
from . import DefaultTable
import array
from collections import namedtuple
import struct
import sys
def _decompileUInt16Array(self, data, offset, numElements, default=0):
    if offset == 0:
        return [default] * numElements
    result = array.array('H', data[offset:offset + 2 * numElements])
    if sys.byteorder != 'big':
        result.byteswap()
    assert len(result) == numElements, result
    return result.tolist()