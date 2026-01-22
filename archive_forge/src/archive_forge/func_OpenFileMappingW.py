import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def OpenFileMappingW(dwDesiredAccess, bInheritHandle, lpName):
    _OpenFileMappingW = windll.kernel32.OpenFileMappingW
    _OpenFileMappingW.argtypes = [DWORD, BOOL, LPWSTR]
    _OpenFileMappingW.restype = HANDLE
    _OpenFileMappingW.errcheck = RaiseIfZero
    hFileMappingObject = _OpenFileMappingW(dwDesiredAccess, bool(bInheritHandle), lpName)
    return FileMappingHandle(hFileMappingObject)