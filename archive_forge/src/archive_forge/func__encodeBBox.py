from io import BytesIO
import sys
import array
import struct
from collections import OrderedDict
from fontTools.misc import sstruct
from fontTools.misc.arrayTools import calcIntBounds
from fontTools.misc.textTools import Tag, bytechr, byteord, bytesjoin, pad
from fontTools.ttLib import (
from fontTools.ttLib.sfnt import (
from fontTools.ttLib.tables import ttProgram, _g_l_y_f
import logging
def _encodeBBox(self, glyphID, glyph):
    assert glyph.numberOfContours != 0, 'empty glyph has no bbox'
    if not glyph.isComposite():
        currentBBox = (glyph.xMin, glyph.yMin, glyph.xMax, glyph.yMax)
        calculatedBBox = calcIntBounds(glyph.coordinates)
        if currentBBox == calculatedBBox:
            return
    self.bboxBitmap[glyphID >> 3] |= 128 >> (glyphID & 7)
    self.bboxStream += sstruct.pack(bboxFormat, glyph)