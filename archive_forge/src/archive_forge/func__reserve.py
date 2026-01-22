from datetime import datetime as _DateTime
import sys
import struct
from .exceptions import BufferFull, OutOfData, ExtraData, FormatError, StackError
from .ext import ExtType, Timestamp
def _reserve(self, n, raise_outofdata=True):
    remain_bytes = len(self._buffer) - self._buff_i - n
    if remain_bytes >= 0:
        return
    if self._feeding:
        self._buff_i = self._buf_checkpoint
        raise OutOfData
    if self._buf_checkpoint > 0:
        del self._buffer[:self._buf_checkpoint]
        self._buff_i -= self._buf_checkpoint
        self._buf_checkpoint = 0
    remain_bytes = -remain_bytes
    if remain_bytes + len(self._buffer) > self._max_buffer_size:
        raise BufferFull
    while remain_bytes > 0:
        to_read_bytes = max(self._read_size, remain_bytes)
        read_data = self.file_like.read(to_read_bytes)
        if not read_data:
            break
        assert isinstance(read_data, bytes)
        self._buffer += read_data
        remain_bytes -= len(read_data)
    if len(self._buffer) < n + self._buff_i and raise_outofdata:
        self._buff_i = 0
        raise OutOfData