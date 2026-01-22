from ctypes import (c_bool, c_char_p, c_int, c_size_t, c_uint, Structure, byref,
from collections import namedtuple
from enum import IntFlag
from llvmlite.binding import ffi
import os
from tempfile import mkstemp
from llvmlite.binding.common import _encode_string
def add_arg_promotion_pass(self, max_elements=3):
    """
        See https://llvm.org/docs/Passes.html#argpromotion-promote-by-reference-arguments-to-scalars

        LLVM 14: `llvm::createArgumentPromotionPass`
        """
    ffi.lib.LLVMPY_AddArgPromotionPass(self, max_elements)