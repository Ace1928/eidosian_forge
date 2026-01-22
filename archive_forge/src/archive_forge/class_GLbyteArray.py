import ctypes
import OpenGL
from OpenGL.raw.GL import _types
from OpenGL import plugins
from OpenGL.arrays import formathandler, _arrayconstants as GL_1_1
from OpenGL import logs
from OpenGL import acceleratesupport
class GLbyteArray(ArrayDatatype, ctypes.POINTER(_types.GLbyte)):
    """Array datatype for GLbyte types"""
    baseType = _types.GLbyte
    typeConstant = _types.GL_BYTE