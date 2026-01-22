from ctypes import (c_bool, c_char_p, c_int, c_size_t, c_uint, Structure, byref,
from collections import namedtuple
from enum import IntFlag
from llvmlite.binding import ffi
import os
from tempfile import mkstemp
from llvmlite.binding.common import _encode_string
def add_sroa_pass(self):
    """
        See http://llvm.org/docs/Passes.html#scalarrepl-scalar-replacement-of-aggregates-dt
        Note that this pass corresponds to the ``opt -sroa`` command-line option,
        despite the link above.

        LLVM 14: `llvm::createSROAPass`
        """
    ffi.lib.LLVMPY_AddSROAPass(self)