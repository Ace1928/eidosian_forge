import sys
from threading import Lock
from typing import Union
from ._cffi_ppmd import ffi, lib
class Ppmd8Encoder(PpmdBaseEncoder):

    def __init__(self, max_order, mem_size, restore_method=PPMD8_RESTORE_METHOD_RESTART):
        self.lock = Lock()
        if mem_size > sys.maxsize:
            raise ValueError('Mem_size exceed to platform limit.')
        self._init_common()
        self.ppmd = ffi.new('CPpmd8 *')
        lib.ppmd8_compress_init(self.ppmd, self.writer)
        lib.Ppmd8_Construct(self.ppmd)
        lib.Ppmd8_Alloc(self.ppmd, mem_size, self._allocator)
        lib.Ppmd8_RangeEnc_Init(self.ppmd)
        lib.Ppmd8_Init(self.ppmd, max_order, restore_method)

    def encode(self, data) -> bytes:
        self.lock.acquire()
        in_buf = self._setup_inBuffer(data)
        out, out_buf = self._setup_outBuffer()
        while lib.ppmd8_compress(self.ppmd, out_buf, in_buf) > 0:
            if out_buf.pos == out_buf.size:
                out.grow(out_buf)
        self.lock.release()
        return out.finish(out_buf)

    def flush(self, endmark=True) -> bytes:
        self.lock.acquire()
        if self.flushed:
            self.lock.release()
            return
        self.flushed = True
        out, out_buf = self._setup_outBuffer()
        if endmark:
            lib.Ppmd8_EncodeSymbol(self.ppmd, -1)
        lib.Ppmd8_RangeEnc_FlushData(self.ppmd)
        res = out.finish(out_buf)
        lib.Ppmd8_Free(self.ppmd, self._allocator)
        ffi.release(self.ppmd)
        self._release()
        self.lock.release()
        return res

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if not self.flushed:
            self.flush()