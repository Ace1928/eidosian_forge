import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def OpenFileMappingA(dwDesiredAccess, bInheritHandle, lpName):
    _OpenFileMappingA = windll.kernel32.OpenFileMappingA
    _OpenFileMappingA.argtypes = [DWORD, BOOL, LPSTR]
    _OpenFileMappingA.restype = HANDLE
    _OpenFileMappingA.errcheck = RaiseIfZero
    hFileMappingObject = _OpenFileMappingA(dwDesiredAccess, bool(bInheritHandle), lpName)
    return FileMappingHandle(hFileMappingObject)