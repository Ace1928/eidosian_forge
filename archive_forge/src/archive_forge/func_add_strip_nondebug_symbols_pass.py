from ctypes import (c_bool, c_char_p, c_int, c_size_t, c_uint, Structure, byref,
from collections import namedtuple
from enum import IntFlag
from llvmlite.binding import ffi
import os
from tempfile import mkstemp
from llvmlite.binding.common import _encode_string
def add_strip_nondebug_symbols_pass(self):
    """
        See https://llvm.org/docs/Passes.html#strip-nondebug-strip-all-symbols-except-dbg-symbols-from-a-module

        LLVM 14: `llvm::createStripNonDebugSymbolsPass`
        """
    ffi.lib.LLVMPY_AddStripNondebugSymbolsPass(self)