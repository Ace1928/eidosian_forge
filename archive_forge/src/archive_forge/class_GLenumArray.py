import ctypes
import OpenGL
from OpenGL.raw.GL import _types
from OpenGL import plugins
from OpenGL.arrays import formathandler, _arrayconstants as GL_1_1
from OpenGL import logs
from OpenGL import acceleratesupport
class GLenumArray(ArrayDatatype, ctypes.POINTER(_types.GLenum)):
    """Array datatype for GLenum types"""
    baseType = _types.GLenum
    typeConstant = _types.GL_UNSIGNED_INT