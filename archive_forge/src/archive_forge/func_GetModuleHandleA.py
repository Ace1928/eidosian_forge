import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def GetModuleHandleA(lpModuleName):
    _GetModuleHandleA = windll.kernel32.GetModuleHandleA
    _GetModuleHandleA.argtypes = [LPSTR]
    _GetModuleHandleA.restype = HMODULE
    hModule = _GetModuleHandleA(lpModuleName)
    if hModule == NULL:
        raise ctypes.WinError()
    return hModule