import os
from ctypes import (POINTER, c_char_p, c_longlong, c_int, c_size_t,
from llvmlite.binding import ffi
from llvmlite.binding.common import _decode_string, _encode_string
def add_analysis_passes(self, pm):
    """
        Register analysis passes for this target machine with a pass manager.
        """
    ffi.lib.LLVMPY_AddAnalysisPasses(self, pm)