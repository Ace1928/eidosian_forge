import zlib
from gitdb.util import byte_ord
import mmap
from itertools import islice
from functools import reduce
from gitdb.const import NULL_BYTE, BYTE_SPACE
from gitdb.utils.encoding import force_text
from gitdb.typ import (
from io import StringIO
class DeltaChunkList(list):
    """List with special functionality to deal with DeltaChunks.
    There are two types of lists we represent. The one was created bottom-up, working
    towards the latest delta, the other kind was created top-down, working from the
    latest delta down to the earliest ancestor. This attribute is queryable
    after all processing with is_reversed."""
    __slots__ = tuple()

    def rbound(self):
        """:return: rightmost extend in bytes, absolute"""
        if len(self) == 0:
            return 0
        return self[-1].rbound()

    def lbound(self):
        """:return: leftmost byte at which this chunklist starts"""
        if len(self) == 0:
            return 0
        return self[0].to

    def size(self):
        """:return: size of bytes as measured by our delta chunks"""
        return self.rbound() - self.lbound()

    def apply(self, bbuf, write):
        """Only used by public clients, internally we only use the global routines
        for performance"""
        return delta_list_apply(self, bbuf, write)

    def compress(self):
        """Alter the list to reduce the amount of nodes. Currently we concatenate
        add-chunks
        :return: self"""
        slen = len(self)
        if slen < 2:
            return self
        i = 0
        first_data_index = None
        while i < slen:
            dc = self[i]
            i += 1
            if dc.data is None:
                if first_data_index is not None and i - 2 - first_data_index > 1:
                    nd = StringIO()
                    so = self[first_data_index].to
                    for x in range(first_data_index, i - 1):
                        xdc = self[x]
                        nd.write(xdc.data[:xdc.ts])
                    del self[first_data_index:i - 1]
                    buf = nd.getvalue()
                    self.insert(first_data_index, DeltaChunk(so, len(buf), 0, buf))
                    slen = len(self)
                    i = first_data_index + 1
                first_data_index = None
                continue
            if first_data_index is None:
                first_data_index = i - 1
        return self

    def check_integrity(self, target_size=-1):
        """Verify the list has non-overlapping chunks only, and the total size matches
        target_size
        :param target_size: if not -1, the total size of the chain must be target_size
        :raise AssertionError: if the size doesn't match"""
        if target_size > -1:
            assert self[-1].rbound() == target_size
            assert reduce(lambda x, y: x + y, (d.ts for d in self), 0) == target_size
        if len(self) < 2:
            return
        for dc in self:
            assert dc.ts > 0
            if dc.has_data():
                assert len(dc.data) >= dc.ts
        left = islice(self, 0, len(self) - 1)
        right = iter(self)
        right.next()
        for lft, rgt in zip(left, right):
            assert lft.rbound() == rgt.to
            assert lft.to + lft.ts == rgt.to