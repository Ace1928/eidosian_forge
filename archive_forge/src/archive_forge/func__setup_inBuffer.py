import sys
from threading import Lock
from typing import Union
from ._cffi_ppmd import ffi, lib
def _setup_inBuffer(self, data):
    in_buf = self.reader.inBuffer
    if self._in_begin == self._in_end:
        use_input_buffer = False
        in_buf.src = ffi.from_buffer(data)
        in_buf.size = len(data)
        in_buf.pos = 0
    elif len(data) == 0:
        assert self._in_begin < self._in_end
        use_input_buffer = True
        in_buf.src = self._input_buffer + self._in_begin
        in_buf.size = self._in_end - self._in_begin
        in_buf.pos = 0
    else:
        use_input_buffer = True
        used_now = self._in_end - self._in_begin
        avail_now = self._input_buffer_size - self._in_end
        avail_total = self._input_buffer_size - used_now
        assert used_now > 0 and avail_now >= 0 and (avail_total >= 0)
        if avail_total < len(data):
            new_size = used_now + len(data)
            tmp = _new_nonzero('char[]', new_size)
            if tmp == ffi.NULL:
                raise MemoryError
            ffi.memmove(tmp, self._input_buffer + self._in_begin, used_now)
            self._input_buffer = tmp
            self._input_buffer_size = new_size
            self._in_begin = 0
            self._in_end = used_now
        elif avail_now < len(data):
            ffi.memmove(self._input_buffer, self._input_buffer + self._in_begin, used_now)
            self._in_begin = 0
            self._in_end = used_now
        ffi.memmove(self._input_buffer + self._in_end, ffi.from_buffer(data), len(data))
        self._in_end += len(data)
        in_buf.src = self._input_buffer + self._in_begin
        in_buf.size = used_now + len(data)
        in_buf.pos = 0
    return (in_buf, use_input_buffer)