from fontTools.misc import sstruct
from fontTools.misc.textTools import (
from .BitmapGlyphMetrics import (
from . import DefaultTable
import itertools
import os
import struct
import logging
def _writeExtFileImageData(strikeIndex, glyphName, bitmapObject, writer, ttFont):
    try:
        folder = os.path.dirname(writer.file.name)
    except AttributeError:
        folder = '.'
    folder = os.path.join(folder, 'bitmaps')
    filename = glyphName + bitmapObject.fileExtension
    if not os.path.isdir(folder):
        os.makedirs(folder)
    folder = os.path.join(folder, 'strike%d' % strikeIndex)
    if not os.path.isdir(folder):
        os.makedirs(folder)
    fullPath = os.path.join(folder, filename)
    writer.simpletag('extfileimagedata', value=fullPath)
    writer.newline()
    with open(fullPath, 'wb') as file:
        file.write(bitmapObject.imageData)