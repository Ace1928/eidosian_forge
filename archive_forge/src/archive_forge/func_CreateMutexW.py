import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def CreateMutexW(lpMutexAttributes=None, bInitialOwner=True, lpName=None):
    _CreateMutexW = windll.kernel32.CreateMutexW
    _CreateMutexW.argtypes = [LPVOID, BOOL, LPWSTR]
    _CreateMutexW.restype = HANDLE
    _CreateMutexW.errcheck = RaiseIfZero
    return Handle(_CreateMutexW(lpMutexAttributes, bInitialOwner, lpName))