from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def StartServiceW(hService, ServiceArgVectors=None):
    _StartServiceW = windll.advapi32.StartServiceW
    _StartServiceW.argtypes = [SC_HANDLE, DWORD, LPVOID]
    _StartServiceW.restype = bool
    _StartServiceW.errcheck = RaiseIfZero
    if ServiceArgVectors:
        dwNumServiceArgs = len(ServiceArgVectors)
        CServiceArgVectors = (LPWSTR * dwNumServiceArgs)(*ServiceArgVectors)
        lpServiceArgVectors = ctypes.pointer(CServiceArgVectors)
    else:
        dwNumServiceArgs = 0
        lpServiceArgVectors = None
    _StartServiceW(hService, dwNumServiceArgs, lpServiceArgVectors)