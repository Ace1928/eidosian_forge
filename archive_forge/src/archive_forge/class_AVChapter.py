from ctypes import c_int, c_int64
from ctypes import c_uint8, c_uint, c_double, c_ubyte, c_size_t, c_char, c_char_p
from ctypes import c_void_p, POINTER, CFUNCTYPE, Structure
import pyglet.lib
from pyglet.util import debug_print
from . import compat
from . import libavcodec
from . import libavutil
class AVChapter(Structure):
    pass