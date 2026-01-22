import ctypes
import functools
from winappdbg import compat
import sys
class LIST_ENTRY(Structure):
    _fields_ = [('Flink', PVOID), ('Blink', PVOID)]