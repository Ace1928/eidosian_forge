import sys
from threading import Lock
from typing import Union
from ._cffi_ppmd import ffi, lib
def _unconsumed_in(self, in_buf, use_input_buffer):
    if in_buf.pos == in_buf.size:
        if use_input_buffer:
            self._in_begin = 0
            self._in_end = 0
    elif in_buf.pos < in_buf.size:
        data_size = in_buf.size - in_buf.pos
        if not use_input_buffer:
            if self._input_buffer == ffi.NULL or self._input_buffer_size < data_size:
                self._input_buffer = _new_nonzero('char[]', data_size)
                if self._input_buffer == ffi.NULL:
                    self._input_buffer_size = 0
                    raise MemoryError
                self._input_buffer_size = data_size
            ffi.memmove(self._input_buffer, in_buf.src + in_buf.pos, data_size)
            self._in_begin = 0
            self._in_end = data_size
        else:
            self._in_begin += in_buf.pos
    else:
        raise PpmdError('Wrong status: input buffer overrun.')