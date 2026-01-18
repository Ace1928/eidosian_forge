from OpenGL import platform as _p, arrays
from OpenGL.raw.WGL import _types as _cs
from OpenGL.raw.WGL._types import *
from OpenGL.raw.WGL import _errors
from OpenGL.constant import Constant as _C
import ctypes
@_f
@_p.types(_cs.HANDLE, _cs.HANDLE, ctypes.c_void_p, _cs.GLuint, _cs.GLenum, _cs.GLenum)
def wglDXRegisterObjectNV(hDevice, dxObject, name, type, access):
    pass