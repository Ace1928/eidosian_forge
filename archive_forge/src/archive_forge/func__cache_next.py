import struct
from aiokafka.errors import CorruptRecordException
from aiokafka.util import NO_EXTENSIONS
from .legacy_records import LegacyRecordBatch
from .default_records import DefaultRecordBatch
def _cache_next(self, len_offset=LENGTH_OFFSET, log_overhead=LOG_OVERHEAD):
    buffer = self._buffer
    buffer_len = len(buffer)
    pos = self._pos
    remaining = buffer_len - pos
    if remaining < log_overhead:
        self._remaining_bytes = remaining
        self._next_slice = None
        return
    length, = struct.unpack_from('>i', buffer, pos + len_offset)
    slice_end = pos + log_overhead + length
    if slice_end > buffer_len:
        self._remaining_bytes = remaining
        self._next_slice = None
        return
    self._next_slice = memoryview(buffer)[pos:slice_end]
    self._pos = slice_end