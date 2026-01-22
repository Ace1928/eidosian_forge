from OpenGL import platform as _p, constant, extensions
from ctypes import *
from OpenGL.raw.GL._types import *
from OpenGL._bytes import as_8_bit
class GLXHyperpipeConfigSGIX(Structure):
    _fields_ = [('pipeName', c_char * 80), ('channel', c_int), ('participationType', c_uint), ('timeSlice', c_int)]