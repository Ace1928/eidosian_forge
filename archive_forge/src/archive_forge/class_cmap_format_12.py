from fontTools.misc.textTools import bytesjoin, safeEval, readHex
from fontTools.misc.encodingTools import getEncoding
from fontTools.ttLib import getSearchRange
from fontTools.unicode import Unicode
from . import DefaultTable
import sys
import struct
import array
import logging
class cmap_format_12(cmap_format_12_or_13):
    _format_step = 1

    def __init__(self, format=12):
        cmap_format_12_or_13.__init__(self, format)

    def _computeGIDs(self, startingGlyph, numberOfGlyphs):
        return list(range(startingGlyph, startingGlyph + numberOfGlyphs))

    def _IsInSameRun(self, glyphID, lastGlyphID, charCode, lastCharCode):
        return glyphID == 1 + lastGlyphID and charCode == 1 + lastCharCode