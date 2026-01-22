from ctypes import (c_bool, c_char_p, c_int, c_size_t, c_uint, Structure, byref,
from collections import namedtuple
from enum import IntFlag
from llvmlite.binding import ffi
import os
from tempfile import mkstemp
from llvmlite.binding.common import _encode_string
def add_break_critical_edges_pass(self):
    """
        See https://llvm.org/docs/Passes.html#break-crit-edges-break-critical-edges-in-cfg

        LLVM 14: `llvm::createBreakCriticalEdgesPass`
        """
    ffi.lib.LLVMPY_AddBreakCriticalEdgesPass(self)