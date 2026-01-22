from ctypes import (c_bool, c_char_p, c_int, c_size_t, c_uint, Structure, byref,
from collections import namedtuple
from enum import IntFlag
from llvmlite.binding import ffi
import os
from tempfile import mkstemp
from llvmlite.binding.common import _encode_string
class RefPruneSubpasses(IntFlag):
    PER_BB = 1
    DIAMOND = 2
    FANOUT = 4
    FANOUT_RAISE = 8
    ALL = PER_BB | DIAMOND | FANOUT | FANOUT_RAISE