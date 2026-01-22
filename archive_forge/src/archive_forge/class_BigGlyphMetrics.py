from fontTools.misc import sstruct
from fontTools.misc.textTools import safeEval
import logging
class BigGlyphMetrics(BitmapGlyphMetrics):
    binaryFormat = bigGlyphMetricsFormat