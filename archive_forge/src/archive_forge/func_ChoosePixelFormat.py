from OpenGL import platform as _p, arrays
from OpenGL.raw.WGL import _types as _cs
from OpenGL.raw.WGL._types import *
from OpenGL.raw.WGL import _errors
from OpenGL.constant import Constant as _C
import ctypes
@_f
@_p.types(_cs.c_int, _cs.HDC, ctypes.POINTER(_cs.PIXELFORMATDESCRIPTOR))
def ChoosePixelFormat(hDc, pPfd):
    pass