import ctypes
from OpenGL.platform import ctypesloader
from OpenGL._bytes import as_8_bit
import sys, logging
from OpenGL import _configflags
from OpenGL import logs, MODULE_ANNOTATIONS
class _CheckContext(object):

    def __init__(self, func, ccisvalid):
        self.func = func
        self.ccisvalid = ccisvalid

    def __setattr__(self, key, value):
        if key not in ('func', 'ccisvalid'):
            return setattr(self.func, key, value)
        else:
            self.__dict__[key] = value

    def __repr__(self):
        if getattr(self.func, '__doc__', None):
            return self.func.__doc__
        else:
            return repr(self.func)

    def __getattr__(self, key):
        if key != 'func':
            return getattr(self.func, key)
        raise AttributeError(key)

    def __call__(self, *args, **named):
        if not self.ccisvalid():
            from OpenGL import error
            raise error.NoContext(self.func.__name__, args, named)
        return self.func(*args, **named)