from ctypes import (c_bool, c_char_p, c_int, c_size_t, c_uint, Structure, byref,
from collections import namedtuple
from enum import IntFlag
from llvmlite.binding import ffi
import os
from tempfile import mkstemp
from llvmlite.binding.common import _encode_string
def add_strip_symbols_pass(self, only_debug=False):
    """
        See https://llvm.org/docs/Passes.html#strip-strip-all-symbols-from-a-module

        LLVM 14: `llvm::createStripSymbolsPass`
        """
    ffi.lib.LLVMPY_AddStripSymbolsPass(self, only_debug)