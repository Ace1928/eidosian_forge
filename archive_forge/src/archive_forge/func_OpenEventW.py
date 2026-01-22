import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def OpenEventW(dwDesiredAccess=EVENT_ALL_ACCESS, bInheritHandle=False, lpName=None):
    _OpenEventW = windll.kernel32.OpenEventW
    _OpenEventW.argtypes = [DWORD, BOOL, LPWSTR]
    _OpenEventW.restype = HANDLE
    _OpenEventW.errcheck = RaiseIfZero
    return Handle(_OpenEventW(dwDesiredAccess, bInheritHandle, lpName))