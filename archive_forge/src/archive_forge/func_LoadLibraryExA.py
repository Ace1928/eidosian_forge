import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def LoadLibraryExA(pszLibrary, dwFlags=0):
    _LoadLibraryExA = windll.kernel32.LoadLibraryExA
    _LoadLibraryExA.argtypes = [LPSTR, HANDLE, DWORD]
    _LoadLibraryExA.restype = HMODULE
    hModule = _LoadLibraryExA(pszLibrary, NULL, dwFlags)
    if hModule == NULL:
        raise ctypes.WinError()
    return hModule