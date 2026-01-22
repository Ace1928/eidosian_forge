import ctypes
import functools
from winappdbg import compat
import sys
class M128A(Structure):
    _fields_ = [('Low', ULONGLONG), ('High', LONGLONG)]