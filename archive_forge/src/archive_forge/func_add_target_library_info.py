from ctypes import (c_bool, c_char_p, c_int, c_size_t, c_uint, Structure, byref,
from collections import namedtuple
from enum import IntFlag
from llvmlite.binding import ffi
import os
from tempfile import mkstemp
from llvmlite.binding.common import _encode_string
def add_target_library_info(self, triple):
    ffi.lib.LLVMPY_AddTargetLibraryInfoPass(self, _encode_string(triple))