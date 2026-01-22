import ctypes
from ctypes import *
import pyglet
from pyglet.gl.lib import missing_function, decorate_function
from pyglet.util import asbytes
class WGLFunctionProxy:
    __slots__ = class_slots

    def __init__(self, name, ftype, requires, suggestions):
        assert _have_get_proc_address
        self.name = name
        self.ftype = ftype
        self.requires = requires
        self.suggestions = suggestions
        self.func = None

    def __call__(self, *args, **kwargs):
        from pyglet.gl import current_context
        if not current_context:
            raise Exception('Call to function "%s" before GL context created' % self.name)
        address = wglGetProcAddress(asbytes(self.name))
        if cast(address, POINTER(c_int)):
            self.func = cast(address, self.ftype)
            decorate_function(self.func, self.name)
        else:
            self.func = missing_function(self.name, self.requires, self.suggestions)
        self.__class__ = makeWGLFunction(self.func)
        return self.func(*args, **kwargs)