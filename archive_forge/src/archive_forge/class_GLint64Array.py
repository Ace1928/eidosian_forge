import ctypes
import OpenGL
from OpenGL.raw.GL import _types
from OpenGL import plugins
from OpenGL.arrays import formathandler, _arrayconstants as GL_1_1
from OpenGL import logs
from OpenGL import acceleratesupport
class GLint64Array(ArrayDatatype, ctypes.POINTER(_types.GLint64)):
    """Array datatype for GLuint types"""
    baseType = _types.GLint64
    typeConstant = None