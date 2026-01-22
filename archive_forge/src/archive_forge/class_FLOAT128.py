import ctypes
import functools
from winappdbg import compat
import sys
class FLOAT128(Structure):
    _fields_ = [('LowPart', QWORD), ('HighPart', QWORD)]