from OpenGL import platform as _p, arrays
from OpenGL.raw.WGL import _types as _cs
from OpenGL.raw.WGL._types import *
from OpenGL.raw.WGL import _errors
from OpenGL.constant import Constant as _C
import ctypes
@_f
@_p.types(ctypes.c_void_p, _cs.GLsizei, _cs.GLfloat, _cs.GLfloat, _cs.GLfloat)
def wglAllocateMemoryNV(size, readfreq, writefreq, priority):
    pass