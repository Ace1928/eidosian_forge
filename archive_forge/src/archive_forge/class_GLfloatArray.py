import ctypes
import OpenGL
from OpenGL.raw.GL import _types
from OpenGL import plugins
from OpenGL.arrays import formathandler, _arrayconstants as GL_1_1
from OpenGL import logs
from OpenGL import acceleratesupport
class GLfloatArray(ArrayDatatype, ctypes.POINTER(_types.GLfloat)):
    """Array datatype for GLfloat types"""
    baseType = _types.GLfloat
    typeConstant = _types.GL_FLOAT