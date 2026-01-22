import os
import six
import sys
from ctypes import cdll
from ctypes import CFUNCTYPE
from ctypes import CDLL
from ctypes import POINTER
from ctypes import Structure
from ctypes import byref
from ctypes import cast
from ctypes import sizeof
from ctypes import py_object
from ctypes import c_char
from ctypes import c_char_p
from ctypes import c_int
from ctypes import c_size_t
from ctypes import c_void_p
from ctypes import memmove
from ctypes.util import find_library
from typing import Union
class PamResponse(Structure):
    """wrapper class for pam_response structure"""
    _fields_ = [('resp', c_char_p), ('resp_retcode', c_int)]

    def __repr__(self):
        return '<PamResponse code: %i, content: %s >' % (self.resp_retcode, self.resp)