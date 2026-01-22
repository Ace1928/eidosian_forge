import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def LoadLibraryA(pszLibrary):
    _LoadLibraryA = windll.kernel32.LoadLibraryA
    _LoadLibraryA.argtypes = [LPSTR]
    _LoadLibraryA.restype = HMODULE
    hModule = _LoadLibraryA(pszLibrary)
    if hModule == NULL:
        raise ctypes.WinError()
    return hModule