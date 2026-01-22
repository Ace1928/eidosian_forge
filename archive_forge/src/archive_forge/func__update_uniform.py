import warnings
from ctypes import *
from weakref import proxy
import pyglet
from pyglet.gl import *
from pyglet.graphics.vertexbuffer import BufferObject
def _update_uniform(self, data, offset=0):
    if offset != 0:
        size = 1
    else:
        size = self._uniform.size
    if self._dsa:
        self._gl_setter(self._uniform.program, self._uniform.location + offset, size, data)
    else:
        glUseProgram(self._uniform.program)
        self._gl_setter(self._uniform.location + offset, size, data)