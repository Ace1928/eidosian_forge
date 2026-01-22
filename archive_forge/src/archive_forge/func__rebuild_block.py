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
def _rebuild_block(self):
    """Create a new GroupCompressBlock with only the referenced texts."""
    compressor = self._make_group_compressor()
    tstart = time.time()
    old_length = self._block._content_length
    end_point = 0
    for factory in self._factories:
        chunks = factory.get_bytes_as('chunked')
        chunks_len = factory.size
        if chunks_len is None:
            chunks_len = sum(map(len, chunks))
        found_sha1, start_point, end_point, type = compressor.compress(factory.key, chunks, chunks_len, factory.sha1)
        factory.sha1 = found_sha1
        factory._start = start_point
        factory._end = end_point
    self._last_byte = end_point
    new_block = compressor.flush()
    delta = time.time() - tstart
    self._block = new_block
    trace.mutter('creating new compressed block on-the-fly in %.3fs %d bytes => %d bytes', delta, old_length, self._block._content_length)