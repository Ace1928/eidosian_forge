import zlib
from gitdb.util import byte_ord
import mmap
from itertools import islice
from functools import reduce
from gitdb.const import NULL_BYTE, BYTE_SPACE
from gitdb.utils.encoding import force_text
from gitdb.typ import (
from io import StringIO
class DeltaChunk:
    """Represents a piece of a delta, it can either add new data, or copy existing
    one from a source buffer"""
    __slots__ = ('to', 'ts', 'so', 'data')

    def __init__(self, to, ts, so, data):
        self.to = to
        self.ts = ts
        self.so = so
        self.data = data

    def __repr__(self):
        return 'DeltaChunk(%i, %i, %s, %s)' % (self.to, self.ts, self.so, self.data or '')

    def rbound(self):
        return self.to + self.ts

    def has_data(self):
        """:return: True if the instance has data to add to the target stream"""
        return self.data is not None