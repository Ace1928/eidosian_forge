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
class Int8(IntValue):
    staticSize = 1

    def read(self, reader, font, tableDict):
        return reader.readInt8()

    def readArray(self, reader, font, tableDict, count):
        return reader.readInt8Array(count)

    def write(self, writer, font, tableDict, value, repeatIndex=None):
        writer.writeInt8(value)

    def writeArray(self, writer, font, tableDict, values):
        writer.writeInt8Array(values)