from ctypes import *
from ctypes import _SimpleCData, _check_size
from OpenGL import extensions
from OpenGL.raw.GL._types import *
from OpenGL._bytes import as_8_bit
from OpenGL._opaque import opaque_pointer_cls as _opaque_pointer_cls
class HANDLE(_SimpleCData):
    """Github Issue #8 CTypes shares all references to c_void_p
    
    We have to have a separate type to avoid short-circuiting all
    of the array-handling machinery for real c_void_p arguments.
    """
    _type_ = 'P'