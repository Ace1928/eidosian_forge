import ctypes
import OpenGL
from OpenGL.raw.GL import _types
from OpenGL import plugins
from OpenGL.arrays import formathandler, _arrayconstants as GL_1_1
from OpenGL import logs
from OpenGL import acceleratesupport
class GLfixedArray(ArrayDatatype, ctypes.POINTER(_types.GLfixed)):
    baseType = _types.GLfixed
    typeConstant = _types.GL_FIXED