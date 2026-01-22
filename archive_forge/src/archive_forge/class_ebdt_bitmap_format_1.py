from fontTools.misc import sstruct
from fontTools.misc.textTools import (
from .BitmapGlyphMetrics import (
from . import DefaultTable
import itertools
import os
import struct
import logging
class ebdt_bitmap_format_1(ByteAlignedBitmapMixin, BitmapPlusSmallMetricsMixin, BitmapGlyph):

    def decompile(self):
        self.metrics = SmallGlyphMetrics()
        dummy, data = sstruct.unpack2(smallGlyphMetricsFormat, self.data, self.metrics)
        self.imageData = data

    def compile(self, ttFont):
        data = sstruct.pack(smallGlyphMetricsFormat, self.metrics)
        return data + self.imageData