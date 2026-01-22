from fontTools.misc.fixedTools import (
from fontTools.misc.roundTools import nearestMultipleShortestRepr, otRound
from fontTools.misc.textTools import bytesjoin, tobytes, tostr, pad, safeEval
from fontTools.ttLib import getSearchRange
from .otBase import (
from .otTables import (
from itertools import zip_longest
from functools import partial
import re
import struct
from typing import Optional
import logging
class _LazyList(UserList):

    def __getslice__(self, i, j):
        return self.__getitem__(slice(i, j))

    def __getitem__(self, k):
        if isinstance(k, slice):
            indices = range(*k.indices(len(self)))
            return [self[i] for i in indices]
        item = self.data[k]
        if isinstance(item, _MissingItem):
            self.reader.seek(self.pos + item[0] * self.recordSize)
            item = self.conv.read(self.reader, self.font, {})
            self.data[k] = item
        return item

    def __add__(self, other):
        if isinstance(other, _LazyList):
            other = list(other)
        elif isinstance(other, list):
            pass
        else:
            return NotImplemented
        return list(self) + other

    def __radd__(self, other):
        if not isinstance(other, list):
            return NotImplemented
        return other + list(self)