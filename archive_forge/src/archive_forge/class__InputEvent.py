import libevdev
import os
import ctypes
import errno
from ctypes import c_char_p
from ctypes import c_int
from ctypes import c_uint
from ctypes import c_void_p
from ctypes import c_long
from ctypes import c_int32
from ctypes import c_uint16
class _InputEvent(ctypes.Structure):
    _fields_ = [('sec', c_long), ('usec', c_long), ('type', c_uint16), ('code', c_uint16), ('value', c_int32)]