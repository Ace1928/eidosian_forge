import ctypes
import OpenGL
from OpenGL.raw.GL import _types
from OpenGL import plugins
from OpenGL.arrays import formathandler, _arrayconstants as GL_1_1
from OpenGL import logs
from OpenGL import acceleratesupport
class GLcharArray(ArrayDatatype, ctypes.c_char_p):
    """Array datatype for ARB extension pointers-to-arrays"""
    baseType = _types.GLchar
    typeConstant = _types.GL_BYTE