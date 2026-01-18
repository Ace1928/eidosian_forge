from OpenGL import platform as _p, arrays
from OpenGL.raw.WGL import _types as _cs
from OpenGL.raw.WGL._types import *
from OpenGL.raw.WGL import _errors
from OpenGL.constant import Constant as _C
import ctypes
@_f
@_p.types(_cs.BOOL, _cs.HDC, _cs.c_int, _cs.c_int, _cs.UINT, ctypes.POINTER(_cs.c_int), ctypes.POINTER(_cs.FLOAT))
def wglGetPixelFormatAttribfvEXT(hdc, iPixelFormat, iLayerPlane, nAttributes, piAttributes, pfValues):
    pass