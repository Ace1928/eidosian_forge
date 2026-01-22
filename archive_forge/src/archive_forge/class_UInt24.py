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
class UInt24(IntValue):
    staticSize = 3

    def read(self, reader, font, tableDict):
        return reader.readUInt24()

    def write(self, writer, font, tableDict, value, repeatIndex=None):
        writer.writeUInt24(value)