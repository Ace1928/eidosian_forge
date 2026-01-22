from ctypes import (c_bool, c_char_p, c_int, c_size_t, c_uint, Structure, byref,
from collections import namedtuple
from enum import IntFlag
from llvmlite.binding import ffi
import os
from tempfile import mkstemp
from llvmlite.binding.common import _encode_string
def add_basic_alias_analysis_pass(self):
    """
        See http://llvm.org/docs/AliasAnalysis.html#the-basicaa-pass

        LLVM 14: `LLVMAddBasicAliasAnalysisPass`
        """
    ffi.lib.LLVMPY_AddBasicAliasAnalysisPass(self)