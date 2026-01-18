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
def writeFormat2(self, writer, font, segments):
    writer.writeUShort(2)
    valueSize = self.converter.staticSize
    numUnits, unitSize = (len(segments), valueSize + 4)
    self.writeBinSearchHeader(writer, numUnits, unitSize)
    for firstGlyph, lastGlyph, value in segments:
        writer.writeUShort(lastGlyph)
        writer.writeUShort(firstGlyph)
        self.converter.write(writer, font, tableDict=None, value=value, repeatIndex=None)
    writer.writeUShort(65535)
    writer.writeUShort(65535)
    writer.writeData(b'\x00' * valueSize)