from fontTools.misc.roundTools import noRound, otRound
from fontTools.misc.intTools import bit_count
from fontTools.ttLib.tables import otTables as ot
from fontTools.varLib.models import supportScalar
from fontTools.varLib.builder import (
from functools import partial
from collections import defaultdict
from heapq import heappush, heappop
class _Encoding(object):

    def __init__(self, chars):
        self.chars = chars
        self.width = bit_count(chars)
        self.columns = self._columns(chars)
        self.overhead = self._characteristic_overhead(self.columns)
        self.items = set()

    def append(self, row):
        self.items.add(row)

    def extend(self, lst):
        self.items.update(lst)

    def get_room(self):
        """Maximum number of bytes that can be added to characteristic
        while still being beneficial to merge it into another one."""
        count = len(self.items)
        return max(0, (self.overhead - 1) // count - self.width)
    room = property(get_room)

    def get_gain(self):
        """Maximum possible byte gain from merging this into another
        characteristic."""
        count = len(self.items)
        return max(0, self.overhead - count)
    gain = property(get_gain)

    def gain_sort_key(self):
        return (self.gain, self.chars)

    def width_sort_key(self):
        return (self.width, self.chars)

    @staticmethod
    def _characteristic_overhead(columns):
        """Returns overhead in bytes of encoding this characteristic
        as a VarData."""
        c = 4 + 6
        c += bit_count(columns) * 2
        return c

    @staticmethod
    def _columns(chars):
        cols = 0
        i = 1
        while chars:
            if chars & 15:
                cols |= i
            chars >>= 4
            i <<= 1
        return cols

    def gain_from_merging(self, other_encoding):
        combined_chars = other_encoding.chars | self.chars
        combined_width = bit_count(combined_chars)
        combined_columns = self.columns | other_encoding.columns
        combined_overhead = _Encoding._characteristic_overhead(combined_columns)
        combined_gain = +self.overhead + other_encoding.overhead - combined_overhead - (combined_width - self.width) * len(self.items) - (combined_width - other_encoding.width) * len(other_encoding.items)
        return combined_gain