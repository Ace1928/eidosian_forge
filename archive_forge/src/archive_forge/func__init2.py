import sys
from threading import Lock
from typing import Union
from ._cffi_ppmd import ffi, lib
def _init2(self):
    lib.ppmd8_decompress_init(self.ppmd, self.reader, self.threadInfo, self._allocator)
    lib.Ppmd8_RangeDec_Init(self.ppmd)