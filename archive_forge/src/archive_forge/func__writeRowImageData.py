from fontTools.misc import sstruct
from fontTools.misc.textTools import (
from .BitmapGlyphMetrics import (
from . import DefaultTable
import itertools
import os
import struct
import logging
def _writeRowImageData(strikeIndex, glyphName, bitmapObject, writer, ttFont):
    metrics = bitmapObject.exportMetrics
    del bitmapObject.exportMetrics
    bitDepth = bitmapObject.exportBitDepth
    del bitmapObject.exportBitDepth
    writer.begintag('rowimagedata', bitDepth=bitDepth, width=metrics.width, height=metrics.height)
    writer.newline()
    for curRow in range(metrics.height):
        rowData = bitmapObject.getRow(curRow, bitDepth=bitDepth, metrics=metrics)
        writer.simpletag('row', value=hexStr(rowData))
        writer.newline()
    writer.endtag('rowimagedata')
    writer.newline()