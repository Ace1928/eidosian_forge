import struct
from aiokafka.errors import CorruptRecordException
from aiokafka.util import NO_EXTENSIONS
from .legacy_records import LegacyRecordBatch
from .default_records import DefaultRecordBatch
class _MemoryRecordsPy:
    LENGTH_OFFSET = struct.calcsize('>q')
    LOG_OVERHEAD = struct.calcsize('>qi')
    MAGIC_OFFSET = struct.calcsize('>qii')
    MIN_SLICE = LOG_OVERHEAD + LegacyRecordBatch.RECORD_OVERHEAD_V0

    def __init__(self, bytes_data):
        self._buffer = bytes_data
        self._pos = 0
        self._next_slice = None
        self._remaining_bytes = 0
        self._cache_next()

    def size_in_bytes(self):
        return len(self._buffer)

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

    def has_next(self):
        return self._next_slice is not None

    def next_batch(self, _min_slice=MIN_SLICE, _magic_offset=MAGIC_OFFSET):
        next_slice = self._next_slice
        if next_slice is None:
            return None
        if len(next_slice) < _min_slice:
            raise CorruptRecordException('Record size is less than the minimum record overhead ({})'.format(_min_slice - self.LOG_OVERHEAD))
        self._cache_next()
        magic = next_slice[_magic_offset]
        if magic >= 2:
            return DefaultRecordBatch(next_slice)
        else:
            return LegacyRecordBatch(next_slice, magic)