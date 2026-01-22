import ctypes
import OpenGL
from OpenGL.raw.GL import _types
from OpenGL import plugins
from OpenGL.arrays import formathandler, _arrayconstants as GL_1_1
from OpenGL import logs
from OpenGL import acceleratesupport
class GLclampfArray(ArrayDatatype, ctypes.POINTER(_types.GLclampf)):
    """Array datatype for GLclampf types"""
    baseType = _types.GLclampf
    typeConstant = _types.GL_FLOAT