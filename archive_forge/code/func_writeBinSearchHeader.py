from fontTools.misc.fixedTools import (
from fontTools.misc.roundTools import nearestMultipleShortestRepr, otRound
from fontTools.misc.textTools import bytesjoin, tobytes, tostr, pad, safeEval
from fontTools.ttLib import getSearchRange
from .otBase import (
from .otTables import (
from itertools import zip_longest
from functools import partial
import re
import struct
from typing import Optional
import logging
@staticmethod
def writeBinSearchHeader(writer, numUnits, unitSize):
    writer.writeUShort(unitSize)
    writer.writeUShort(numUnits)
    searchRange, entrySelector, rangeShift = getSearchRange(n=numUnits, itemSize=unitSize)
    writer.writeUShort(searchRange)
    writer.writeUShort(entrySelector)
    writer.writeUShort(rangeShift)