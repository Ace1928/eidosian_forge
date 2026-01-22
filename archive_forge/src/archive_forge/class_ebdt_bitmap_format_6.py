from fontTools.misc import sstruct
from fontTools.misc.textTools import (
from .BitmapGlyphMetrics import (
from . import DefaultTable
import itertools
import os
import struct
import logging
class ebdt_bitmap_format_6(ByteAlignedBitmapMixin, BitmapPlusBigMetricsMixin, BitmapGlyph):

    def decompile(self):
        self.metrics = BigGlyphMetrics()
        dummy, data = sstruct.unpack2(bigGlyphMetricsFormat, self.data, self.metrics)
        self.imageData = data

    def compile(self, ttFont):
        data = sstruct.pack(bigGlyphMetricsFormat, self.metrics)
        return data + self.imageData