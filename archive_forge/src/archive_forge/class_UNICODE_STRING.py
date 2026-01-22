import ctypes
import functools
from winappdbg import compat
import sys
class UNICODE_STRING(Structure):
    _fields_ = [('Length', USHORT), ('MaximumLength', USHORT), ('Buffer', PVOID)]