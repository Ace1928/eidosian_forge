import sys
from threading import Lock
from typing import Union
from ._cffi_ppmd import ffi, lib
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