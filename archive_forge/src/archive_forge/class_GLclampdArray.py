import ctypes
import OpenGL
from OpenGL.raw.GL import _types
from OpenGL import plugins
from OpenGL.arrays import formathandler, _arrayconstants as GL_1_1
from OpenGL import logs
from OpenGL import acceleratesupport
class GLclampdArray(ArrayDatatype, ctypes.POINTER(_types.GLclampd)):
    """Array datatype for GLclampd types"""
    baseType = _types.GLclampd
    typeConstant = _types.GL_DOUBLE