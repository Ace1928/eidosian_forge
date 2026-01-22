from fontTools.misc import sstruct
from fontTools.misc.textTools import (
from .BitmapGlyphMetrics import (
from . import DefaultTable
import itertools
import os
import struct
import logging
def _data2binary(data, numBits):
    binaryList = []
    for curByte in data:
        value = byteord(curByte)
        numBitsCut = min(8, numBits)
        for i in range(numBitsCut):
            if value & 1:
                binaryList.append('1')
            else:
                binaryList.append('0')
            value = value >> 1
        numBits -= numBitsCut
    return strjoin(binaryList)