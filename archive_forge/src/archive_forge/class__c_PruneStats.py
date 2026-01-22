from ctypes import (c_bool, c_char_p, c_int, c_size_t, c_uint, Structure, byref,
from collections import namedtuple
from enum import IntFlag
from llvmlite.binding import ffi
import os
from tempfile import mkstemp
from llvmlite.binding.common import _encode_string
class _c_PruneStats(Structure):
    _fields_ = [('basicblock', c_size_t), ('diamond', c_size_t), ('fanout', c_size_t), ('fanout_raise', c_size_t)]