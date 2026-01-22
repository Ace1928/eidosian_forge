from ctypes import c_int, c_uint16, c_int64, c_uint32, c_uint64
from ctypes import c_uint8, c_uint, c_float, c_char_p
from ctypes import c_void_p, POINTER, CFUNCTYPE, Structure
import pyglet.lib
from pyglet.util import debug_print
from . import compat
from . import libavutil
class AVProfile(Structure):
    _fields_ = [('profile', c_int), ('name', c_char_p)]