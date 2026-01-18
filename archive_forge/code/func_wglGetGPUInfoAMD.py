from OpenGL import platform as _p, arrays
from OpenGL.raw.WGL import _types as _cs
from OpenGL.raw.WGL._types import *
from OpenGL.raw.WGL import _errors
from OpenGL.constant import Constant as _C
import ctypes
@_f
@_p.types(_cs.INT, _cs.UINT, _cs.INT, _cs.GLenum, _cs.UINT, ctypes.c_void_p)
def wglGetGPUInfoAMD(id, property, dataType, size, data):
    pass