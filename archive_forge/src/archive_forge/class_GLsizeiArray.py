import ctypes
import OpenGL
from OpenGL.raw.GL import _types
from OpenGL import plugins
from OpenGL.arrays import formathandler, _arrayconstants as GL_1_1
from OpenGL import logs
from OpenGL import acceleratesupport
class GLsizeiArray(ArrayDatatype, ctypes.POINTER(_types.GLsizei)):
    """Array datatype for GLsizei types"""
    baseType = _types.GLsizei
    typeConstant = _types.GL_INT