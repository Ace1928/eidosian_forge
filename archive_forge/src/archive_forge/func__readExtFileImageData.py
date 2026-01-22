from fontTools.misc import sstruct
from fontTools.misc.textTools import (
from .BitmapGlyphMetrics import (
from . import DefaultTable
import itertools
import os
import struct
import logging
def _readExtFileImageData(bitmapObject, name, attrs, content, ttFont):
    fullPath = attrs['value']
    with open(fullPath, 'rb') as file:
        bitmapObject.imageData = file.read()