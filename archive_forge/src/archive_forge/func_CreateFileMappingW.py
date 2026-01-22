import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def CreateFileMappingW(hFile, lpAttributes=None, flProtect=PAGE_EXECUTE_READWRITE, dwMaximumSizeHigh=0, dwMaximumSizeLow=0, lpName=None):
    _CreateFileMappingW = windll.kernel32.CreateFileMappingW
    _CreateFileMappingW.argtypes = [HANDLE, LPVOID, DWORD, DWORD, DWORD, LPWSTR]
    _CreateFileMappingW.restype = HANDLE
    _CreateFileMappingW.errcheck = RaiseIfZero
    if lpAttributes:
        lpAttributes = ctypes.pointer(lpAttributes)
    if not lpName:
        lpName = None
    hFileMappingObject = _CreateFileMappingW(hFile, lpAttributes, flProtect, dwMaximumSizeHigh, dwMaximumSizeLow, lpName)
    return FileMappingHandle(hFileMappingObject)