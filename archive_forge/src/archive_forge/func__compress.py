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
def _compress(self, key, chunks, input_len, max_delta_size, soft=False):
    """see _CommonGroupCompressor._compress"""
    if self._delta_index._source_offset != self.endpoint:
        raise AssertionError('_source_offset != endpoint somehow the DeltaIndex got out of sync with the output lines')
    bytes = b''.join(chunks)
    delta = self._delta_index.make_delta(bytes, max_delta_size)
    if delta is None:
        type = 'fulltext'
        enc_length = encode_base128_int(input_len)
        len_mini_header = 1 + len(enc_length)
        self._delta_index.add_source(bytes, len_mini_header)
        new_chunks = [b'f', enc_length] + chunks
    else:
        type = 'delta'
        enc_length = encode_base128_int(len(delta))
        len_mini_header = 1 + len(enc_length)
        new_chunks = [b'd', enc_length, delta]
        self._delta_index.add_delta_source(delta, len_mini_header)
    start = self.endpoint
    chunk_start = len(self.chunks)
    self._output_chunks(new_chunks)
    self.input_bytes += input_len
    chunk_end = len(self.chunks)
    self.labels_deltas[key] = (start, chunk_start, self.endpoint, chunk_end)
    if not self._delta_index._source_offset == self.endpoint:
        raise AssertionError('the delta index is out of syncwith the output lines %s != %s' % (self._delta_index._source_offset, self.endpoint))
    return (start, self.endpoint, type)