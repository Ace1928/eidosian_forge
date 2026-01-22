from fontTools.misc import sstruct
from fontTools.misc.roundTools import otRound
from fontTools.misc.textTools import safeEval, num2binary, binary2num
from fontTools.ttLib.tables import DefaultTable
import bisect
import logging
def _getUnicodeRanges():
    if not _unicodeStarts:
        unicodeRanges = [(start, (stop, bit)) for bit, blocks in enumerate(OS2_UNICODE_RANGES) for _, (start, stop) in blocks]
        for start, (stop, bit) in sorted(unicodeRanges):
            _unicodeStarts.append(start)
            _unicodeValues.append((stop, bit))
    return (_unicodeStarts, _unicodeValues)