from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlValue_t(Union):
    _fields_ = [('dVal', c_double), ('uiVal', c_uint), ('ulVal', c_ulong), ('ullVal', c_ulonglong), ('sllVal', c_longlong), ('siVal', c_int)]