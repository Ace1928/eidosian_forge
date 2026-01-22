import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
class EXCEPTION_RECORD32(Structure):
    _fields_ = [('ExceptionCode', DWORD), ('ExceptionFlags', DWORD), ('ExceptionRecord', DWORD), ('ExceptionAddress', DWORD), ('NumberParameters', DWORD), ('ExceptionInformation', DWORD * EXCEPTION_MAXIMUM_PARAMETERS)]