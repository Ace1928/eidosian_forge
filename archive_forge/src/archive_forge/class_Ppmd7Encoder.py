import sys
from threading import Lock
from typing import Union
from ._cffi_ppmd import ffi, lib
class Ppmd7Encoder(PpmdBaseEncoder):

    def __init__(self, max_order: int, mem_size: int):
        if mem_size > sys.maxsize:
            raise ValueError('Mem_size exceed to platform limit.')
        if (_PPMD7_MIN_ORDER > max_order or max_order > _PPMD7_MAX_ORDER) or (_PPMD7_MIN_MEM_SIZE > mem_size or mem_size > _PPMD7_MAX_MEM_SIZE):
            raise ValueError('PPMd wrong parameters.')
        self._init_common()
        self.ppmd = ffi.new('CPpmd7 *')
        self.rc = ffi.new('CPpmd7z_RangeEnc *')
        lib.ppmd7_state_init(self.ppmd, max_order, mem_size, self._allocator)
        lib.ppmd7_compress_init(self.rc, self.writer)

    def encode(self, data) -> bytes:
        self.lock.acquire()
        in_buf = self._setup_inBuffer(data)
        out, out_buf = self._setup_outBuffer()
        while True:
            if lib.ppmd7_compress(self.ppmd, self.rc, out_buf, in_buf) == 0:
                break
            if out_buf.pos == out_buf.size:
                out.grow(out_buf)
        self.lock.release()
        return out.finish(out_buf)

    def flush(self, *, endmark=False) -> bytes:
        if self.flushed:
            raise 'Ppmd7Encoder: Double flush error.'
        self.lock.acquire()
        self.flushed = True
        out, out_buf = self._setup_outBuffer()
        lib.ppmd7_compress_flush(self.ppmd, self.rc, endmark)
        res = out.finish(out_buf)
        lib.ppmd7_state_close(self.ppmd, self._allocator)
        ffi.release(self.ppmd)
        self._release()
        ffi.release(self.rc)
        self.lock.release()
        return res

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if not self.flushed:
            self.flush()