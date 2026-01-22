import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def LoadLibraryExW(pszLibrary, dwFlags=0):
    _LoadLibraryExW = windll.kernel32.LoadLibraryExW
    _LoadLibraryExW.argtypes = [LPWSTR, HANDLE, DWORD]
    _LoadLibraryExW.restype = HMODULE
    hModule = _LoadLibraryExW(pszLibrary, NULL, dwFlags)
    if hModule == NULL:
        raise ctypes.WinError()
    return hModule