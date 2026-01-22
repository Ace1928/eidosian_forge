from fontTools.misc import sstruct
from fontTools.misc.textTools import (
from .BitmapGlyphMetrics import (
from . import DefaultTable
import itertools
import os
import struct
import logging
def _writeRawImageData(strikeIndex, glyphName, bitmapObject, writer, ttFont):
    writer.begintag('rawimagedata')
    writer.newline()
    writer.dumphex(bitmapObject.imageData)
    writer.endtag('rawimagedata')
    writer.newline()