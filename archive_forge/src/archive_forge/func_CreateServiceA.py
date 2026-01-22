from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def CreateServiceA(hSCManager, lpServiceName, lpDisplayName=None, dwDesiredAccess=SERVICE_ALL_ACCESS, dwServiceType=SERVICE_WIN32_OWN_PROCESS, dwStartType=SERVICE_DEMAND_START, dwErrorControl=SERVICE_ERROR_NORMAL, lpBinaryPathName=None, lpLoadOrderGroup=None, lpDependencies=None, lpServiceStartName=None, lpPassword=None):
    _CreateServiceA = windll.advapi32.CreateServiceA
    _CreateServiceA.argtypes = [SC_HANDLE, LPSTR, LPSTR, DWORD, DWORD, DWORD, DWORD, LPSTR, LPSTR, LPDWORD, LPSTR, LPSTR, LPSTR]
    _CreateServiceA.restype = SC_HANDLE
    _CreateServiceA.errcheck = RaiseIfZero
    dwTagId = DWORD(0)
    hService = _CreateServiceA(hSCManager, lpServiceName, dwDesiredAccess, dwServiceType, dwStartType, dwErrorControl, lpBinaryPathName, lpLoadOrderGroup, byref(dwTagId), lpDependencies, lpServiceStartName, lpPassword)
    return (ServiceHandle(hService), dwTagId.value)