from fontTools.misc import sstruct
from fontTools.misc.roundTools import otRound
from fontTools.misc.textTools import safeEval, num2binary, binary2num
from fontTools.ttLib.tables import DefaultTable
import bisect
import logging
def updateFirstAndLastCharIndex(self, ttFont):
    if 'cmap' not in ttFont:
        return
    codes = set()
    for table in getattr(ttFont['cmap'], 'tables', []):
        if table.isUnicode():
            codes.update(table.cmap.keys())
    if codes:
        minCode = min(codes)
        maxCode = max(codes)
        self.usFirstCharIndex = min(65535, minCode)
        self.usLastCharIndex = min(65535, maxCode)