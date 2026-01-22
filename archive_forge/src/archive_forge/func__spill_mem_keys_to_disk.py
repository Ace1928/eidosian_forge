from io import BytesIO
from ..lazy_import import lazy_import
import bisect
import math
import tempfile
import zlib
from .. import (chunk_writer, debug, fifo_cache, lru_cache, osutils, trace,
from . import index, static_tuple
from .index import _OPTION_KEY_ELEMENTS, _OPTION_LEN, _OPTION_NODE_REFS
def _spill_mem_keys_to_disk(self):
    """Write the in memory keys down to disk to cap memory consumption.

        If we already have some keys written to disk, we will combine them so
        as to preserve the sorted order.  The algorithm for combining uses
        powers of two.  So on the first spill, write all mem nodes into a
        single index. On the second spill, combine the mem nodes with the nodes
        on disk to create a 2x sized disk index and get rid of the first index.
        On the third spill, create a single new disk index, which will contain
        the mem nodes, and preserve the existing 2x sized index.  On the fourth,
        combine mem with the first and second indexes, creating a new one of
        size 4x. On the fifth create a single new one, etc.
        """
    if self._combine_backing_indices:
        new_backing_file, size, backing_pos = self._spill_mem_keys_and_combine()
    else:
        new_backing_file, size = self._spill_mem_keys_without_combining()
    new_backing = BTreeGraphIndex(transport.get_transport_from_path('.'), '<temp>', size)
    new_backing._file = new_backing_file
    if self._combine_backing_indices:
        if len(self._backing_indices) == backing_pos:
            self._backing_indices.append(None)
        self._backing_indices[backing_pos] = new_backing
        for backing_pos in range(backing_pos):
            self._backing_indices[backing_pos] = None
    else:
        self._backing_indices.append(new_backing)
    self._nodes = {}
    self._nodes_by_key = None