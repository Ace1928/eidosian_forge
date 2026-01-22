from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def StartServiceA(hService, ServiceArgVectors=None):
    _StartServiceA = windll.advapi32.StartServiceA
    _StartServiceA.argtypes = [SC_HANDLE, DWORD, LPVOID]
    _StartServiceA.restype = bool
    _StartServiceA.errcheck = RaiseIfZero
    if ServiceArgVectors:
        dwNumServiceArgs = len(ServiceArgVectors)
        CServiceArgVectors = (LPSTR * dwNumServiceArgs)(*ServiceArgVectors)
        lpServiceArgVectors = ctypes.pointer(CServiceArgVectors)
    else:
        dwNumServiceArgs = 0
        lpServiceArgVectors = None
    _StartServiceA(hService, dwNumServiceArgs, lpServiceArgVectors)