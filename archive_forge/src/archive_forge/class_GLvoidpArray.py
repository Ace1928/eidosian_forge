import ctypes
import OpenGL
from OpenGL.raw.GL import _types
from OpenGL import plugins
from OpenGL.arrays import formathandler, _arrayconstants as GL_1_1
from OpenGL import logs
from OpenGL import acceleratesupport
class GLvoidpArray(ArrayDatatype, ctypes.POINTER(_types.GLvoid)):
    """Array datatype for GLenum types"""
    baseType = _types.GLvoidp
    typeConstant = _types.GL_VOID_P