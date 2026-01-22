from fontTools.misc import sstruct
from fontTools.misc.textTools import safeEval
import logging
class SmallGlyphMetrics(BitmapGlyphMetrics):
    binaryFormat = smallGlyphMetricsFormat