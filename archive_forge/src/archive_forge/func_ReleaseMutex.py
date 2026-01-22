import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def ReleaseMutex(hMutex):
    _ReleaseMutex = windll.kernel32.ReleaseMutex
    _ReleaseMutex.argtypes = [HANDLE]
    _ReleaseMutex.restype = bool
    _ReleaseMutex.errcheck = RaiseIfZero
    _ReleaseMutex(hMutex)