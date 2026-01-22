import ctypes
import OpenGL
from OpenGL.raw.GL import _types
from OpenGL import plugins
from OpenGL.arrays import formathandler, _arrayconstants as GL_1_1
from OpenGL import logs
from OpenGL import acceleratesupport
class GLdoubleArray(ArrayDatatype, ctypes.POINTER(_types.GLdouble)):
    """Array datatype for GLdouble types"""
    baseType = _types.GLdouble
    typeConstant = _types.GL_DOUBLE