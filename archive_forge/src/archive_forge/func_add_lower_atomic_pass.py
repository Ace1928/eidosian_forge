from ctypes import (c_bool, c_char_p, c_int, c_size_t, c_uint, Structure, byref,
from collections import namedtuple
from enum import IntFlag
from llvmlite.binding import ffi
import os
from tempfile import mkstemp
from llvmlite.binding.common import _encode_string
def add_lower_atomic_pass(self):
    """
        See https://llvm.org/docs/Passes.html#loweratomic-lower-atomic-intrinsics-to-non-atomic-form

        LLVM 14: `llvm::createLowerAtomicPass`
        """
    ffi.lib.LLVMPY_AddLowerAtomicPass(self)