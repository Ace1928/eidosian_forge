from ctypes import (c_bool, c_char_p, c_int, c_size_t, c_uint, Structure, byref,
from collections import namedtuple
from enum import IntFlag
from llvmlite.binding import ffi
import os
from tempfile import mkstemp
from llvmlite.binding.common import _encode_string
def add_always_inliner_pass(self, insert_lifetime=True):
    """
        See https://llvm.org/docs/Passes.html#always-inline-inliner-for-always-inline-functions

        LLVM 14: `llvm::createAlwaysInlinerLegacyPass`
        """
    ffi.lib.LLVMPY_AddAlwaysInlinerPass(self, insert_lifetime)