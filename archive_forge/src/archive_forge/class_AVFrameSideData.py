from ctypes import c_char_p, c_void_p, POINTER, Structure
from ctypes import c_int, c_int64, c_uint64
from ctypes import c_uint8, c_int8, c_uint, c_size_t
import pyglet.lib
from pyglet.util import debug_print
from . import compat
class AVFrameSideData(Structure):
    pass