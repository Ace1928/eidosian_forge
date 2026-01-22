import ctypes
import OpenGL
from OpenGL.raw.GL import _types
from OpenGL import plugins
from OpenGL.arrays import formathandler, _arrayconstants as GL_1_1
from OpenGL import logs
from OpenGL import acceleratesupport
class GLuintArray(ArrayDatatype, ctypes.POINTER(_types.GLuint)):
    """Array datatype for GLuint types"""
    baseType = _types.GLuint
    typeConstant = _types.GL_UNSIGNED_INT