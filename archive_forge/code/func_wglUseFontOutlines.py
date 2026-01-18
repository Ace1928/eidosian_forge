from OpenGL import platform as _p, arrays
from OpenGL.raw.WGL import _types as _cs
from OpenGL.raw.WGL._types import *
from OpenGL.raw.WGL import _errors
from OpenGL.constant import Constant as _C
import ctypes
@_f
@_p.types(_cs.BOOL, _cs.HDC, _cs.DWORD, _cs.DWORD, _cs.DWORD, _cs.FLOAT, _cs.FLOAT, _cs.c_int, _cs.LPGLYPHMETRICSFLOAT)
def wglUseFontOutlines(hDC, first, count, listBase, deviation, extrusion, format, lpgmf):
    pass