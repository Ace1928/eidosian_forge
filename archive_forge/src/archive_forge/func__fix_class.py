import ctypes, ctypes.util, operator, sys
from . import model
@classmethod
def _fix_class(cls):
    cls.__name__ = 'CData<%s>' % (cls._get_c_name(),)
    cls.__qualname__ = 'CData<%s>' % (cls._get_c_name(),)
    cls.__module__ = 'ffi'