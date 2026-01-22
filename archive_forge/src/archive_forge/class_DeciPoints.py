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
class DeciPoints(FloatValue):
    staticSize = 2

    def read(self, reader, font, tableDict):
        return reader.readUShort() / 10

    def write(self, writer, font, tableDict, value, repeatIndex=None):
        writer.writeUShort(round(value * 10))