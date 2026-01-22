from fontTools.misc import sstruct
from fontTools.misc.textTools import (
from .BitmapGlyphMetrics import (
from . import DefaultTable
import itertools
import os
import struct
import logging
def _getByteRange(self, row, bitDepth, metrics):
    rowBytes = (bitDepth * metrics.width + 7) // 8
    byteOffset = row * rowBytes
    return (byteOffset, byteOffset + rowBytes)