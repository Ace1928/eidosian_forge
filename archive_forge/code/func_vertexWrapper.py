from OpenGL.raw import GLU as _simple
from OpenGL.raw.GL.VERSION import GL_1_1
from OpenGL.platform import createBaseFunction
from OpenGL.GLU import glustruct
from OpenGL import arrays, wrapper
from OpenGL.platform import PLATFORM
from OpenGL.lazywrapper import lazy as _lazy
import ctypes
def vertexWrapper(self, function):
    """Converts a vertex-pointer into an OOR vertex for processing"""
    if function is not None and (not hasattr(function, '__call__')):
        raise TypeError('Require a callable callback, got:  %s' % (function,))

    def wrap(vertex, data=None):
        """Just return the original object for polygon_data"""
        vertex = self.originalObject(vertex)
        try:
            if data is not None:
                data = self.originalObject(data)
                return function(vertex, data)
            else:
                return function(vertex)
        except Exception as err:
            err.args += (function, (vertex, data))
            raise
    return wrap