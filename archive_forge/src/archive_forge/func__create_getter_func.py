import warnings
from ctypes import *
from weakref import proxy
import pyglet
from pyglet.gl import *
from pyglet.graphics.vertexbuffer import BufferObject
@staticmethod
def _create_getter_func(program, location, gl_getter, c_array, length):
    """Factory function for creating simplified Uniform getters"""
    if length == 1:

        def getter_func():
            gl_getter(program, location, c_array)
            return c_array[0]
    else:

        def getter_func():
            gl_getter(program, location, c_array)
            return c_array[:]
    return getter_func