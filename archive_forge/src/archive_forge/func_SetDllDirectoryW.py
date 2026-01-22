import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def SetDllDirectoryW(lpPathName):
    _SetDllDirectoryW = windll.kernel32.SetDllDirectoryW
    _SetDllDirectoryW.argytpes = [LPWSTR]
    _SetDllDirectoryW.restype = bool
    _SetDllDirectoryW.errcheck = RaiseIfZero
    _SetDllDirectoryW(lpPathName)