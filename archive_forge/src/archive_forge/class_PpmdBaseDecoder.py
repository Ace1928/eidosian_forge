import sys
from threading import Lock
from typing import Union
from ._cffi_ppmd import ffi, lib
class PpmdBaseDecoder:

    def __init__(self):
        pass

    def _init_common(self):
        self.lock = Lock()
        self._allocator = ffi.new('IAlloc *')
        self._allocator.Alloc = lib.raw_alloc
        self._allocator.Free = lib.raw_free
        self.reader = ffi.new('BufferReader *')
        self._in_buf = _new_nonzero('InBuffer *')
        self.reader.inBuffer = self._in_buf
        self._input_buffer = ffi.NULL
        self._input_buffer_size = 0
        self._in_begin = 0
        self._in_end = 0
        self.closed = False
        self.inited = False

    def _release(self):
        ffi.release(self._in_buf)
        ffi.release(self.reader)
        ffi.release(self._allocator)

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

    def _setup_outBuffer(self):
        out_buf = _new_nonzero('OutBuffer *')
        if out_buf == ffi.NULL:
            raise MemoryError
        out = _BlocksOutputBuffer()
        out.initAndGrow(out_buf, -1)
        return (out, out_buf)

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