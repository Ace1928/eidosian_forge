import time
import zlib
from typing import Type
from ..lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
from breezy.i18n import gettext
from .. import errors, osutils, trace
from ..lru_cache import LRUSizeCache
from .btree_index import BTreeBuilder
from .versionedfile import (AbsentContentFactory, ChunkedContentFactory,
from ._groupcompress_py import (LinesDeltaIndex, apply_delta,
class _LazyGroupCompressFactory:
    """Yield content from a GroupCompressBlock on demand."""

    def __init__(self, key, parents, manager, start, end, first):
        """Create a _LazyGroupCompressFactory

        :param key: The key of just this record
        :param parents: The parents of this key (possibly None)
        :param gc_block: A GroupCompressBlock object
        :param start: Offset of the first byte for this record in the
            uncompressd content
        :param end: Offset of the byte just after the end of this record
            (ie, bytes = content[start:end])
        :param first: Is this the first Factory for the given block?
        """
        self.key = key
        self.parents = parents
        self.sha1 = None
        self.size = None
        self._manager = manager
        self._chunks = None
        self.storage_kind = 'groupcompress-block'
        if not first:
            self.storage_kind = 'groupcompress-block-ref'
        self._first = first
        self._start = start
        self._end = end

    def __repr__(self):
        return '{}({}, first={})'.format(self.__class__.__name__, self.key, self._first)

    def _extract_bytes(self):
        try:
            self._manager._prepare_for_extract()
        except zlib.error as value:
            raise DecompressCorruption('zlib: ' + str(value))
        block = self._manager._block
        self._chunks = block.extract(self.key, self._start, self._end)

    def get_bytes_as(self, storage_kind):
        if storage_kind == self.storage_kind:
            if self._first:
                return self._manager._wire_bytes()
            else:
                return b''
        if storage_kind in ('fulltext', 'chunked', 'lines'):
            if self._chunks is None:
                self._extract_bytes()
            if storage_kind == 'fulltext':
                return b''.join(self._chunks)
            elif storage_kind == 'chunked':
                return self._chunks
            else:
                return osutils.chunks_to_lines(self._chunks)
        raise UnavailableRepresentation(self.key, storage_kind, self.storage_kind)

    def iter_bytes_as(self, storage_kind):
        if self._chunks is None:
            self._extract_bytes()
        if storage_kind == 'chunked':
            return iter(self._chunks)
        elif storage_kind == 'lines':
            return iter(osutils.chunks_to_lines(self._chunks))
        raise UnavailableRepresentation(self.key, storage_kind, self.storage_kind)