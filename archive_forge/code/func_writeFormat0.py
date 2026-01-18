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
def writeFormat0(self, writer, font, values):
    writer.writeUShort(0)
    for glyphID_, value in values:
        self.converter.write(writer, font, tableDict=None, value=value, repeatIndex=None)