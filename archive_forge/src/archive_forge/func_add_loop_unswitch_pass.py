from ctypes import (c_bool, c_char_p, c_int, c_size_t, c_uint, Structure, byref,
from collections import namedtuple
from enum import IntFlag
from llvmlite.binding import ffi
import os
from tempfile import mkstemp
from llvmlite.binding.common import _encode_string
def add_loop_unswitch_pass(self, optimize_for_size=False, has_branch_divergence=False):
    """
        See https://llvm.org/docs/Passes.html#loop-unswitch-unswitch-loops

        LLVM 14: `llvm::createLoopUnswitchPass`
        """
    ffi.lib.LLVMPY_AddLoopUnswitchPass(self, optimize_for_size, has_branch_divergence)