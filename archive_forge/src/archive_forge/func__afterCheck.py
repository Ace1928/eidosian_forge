from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GL import _types, _glgets
from OpenGL.raw.GL.VERSION.GL_2_0 import *
from OpenGL.raw.GL.VERSION.GL_2_0 import _EXTENSION_NAME
import OpenGL
from OpenGL import _configflags
from OpenGL._bytes import bytes, _NULL_8_BYTE, as_8_bit
from OpenGL.raw.GL.ARB.shader_objects import GL_OBJECT_COMPILE_STATUS_ARB as GL_OBJECT_COMPILE_STATUS
from OpenGL.raw.GL.ARB.shader_objects import GL_OBJECT_LINK_STATUS_ARB as GL_OBJECT_LINK_STATUS
from OpenGL.raw.GL.ARB.shader_objects import GL_OBJECT_ACTIVE_UNIFORMS_ARB as GL_OBJECT_ACTIVE_UNIFORMS
from OpenGL.raw.GL.ARB.shader_objects import GL_OBJECT_ACTIVE_UNIFORM_MAX_LENGTH_ARB as GL_OBJECT_ACTIVE_UNIFORM_MAX_LENGTH
from OpenGL.lazywrapper import lazy as _lazy
from OpenGL.raw.GL import _errors
from OpenGL import converters, error, contextdata
from OpenGL.arrays.arraydatatype import ArrayDatatype, GLenumArray
def _afterCheck(key):
    """Generate an error-checking function for compilation operations"""
    if key == GL_OBJECT_COMPILE_STATUS:
        getter = glGetShaderiv
    else:
        getter = glGetProgramiv

    def GLSLCheckError(result, baseOperation=None, cArguments=None, *args):
        result = _errors._error_checker.glCheckError(result, baseOperation, cArguments, *args)
        status = ctypes.c_int()
        getter(cArguments[0], key, ctypes.byref(status))
        status = status.value
        if not status:
            raise error.GLError(result=result, baseOperation=baseOperation, cArguments=cArguments, description=glGetShaderInfoLog(cArguments[0]))
        return result
    return GLSLCheckError