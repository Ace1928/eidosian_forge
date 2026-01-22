from fontTools.misc.roundTools import noRound, otRound
from fontTools.misc.intTools import bit_count
from fontTools.ttLib.tables import otTables as ot
from fontTools.varLib.models import supportScalar
from fontTools.varLib.builder import (
from functools import partial
from collections import defaultdict
from heapq import heappush, heappop
@staticmethod
def _row_characteristics(row):
    """Returns encoding characteristics for a row."""
    longWords = False
    chars = 0
    i = 1
    for v in row:
        if v:
            chars += i
        if not -128 <= v <= 127:
            chars += i * 2
        if not -32768 <= v <= 32767:
            longWords = True
            break
        i <<= 4
    if longWords:
        chars = 0
        i = 1
        for v in row:
            if v:
                chars += i * 3
            if not -32768 <= v <= 32767:
                chars += i * 12
            i <<= 4
    return chars