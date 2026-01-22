from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def QueryServiceStatusEx(hService, InfoLevel=SC_STATUS_PROCESS_INFO):
    if InfoLevel != SC_STATUS_PROCESS_INFO:
        raise NotImplementedError()
    _QueryServiceStatusEx = windll.advapi32.QueryServiceStatusEx
    _QueryServiceStatusEx.argtypes = [SC_HANDLE, SC_STATUS_TYPE, LPVOID, DWORD, LPDWORD]
    _QueryServiceStatusEx.restype = bool
    _QueryServiceStatusEx.errcheck = RaiseIfZero
    lpBuffer = SERVICE_STATUS_PROCESS()
    cbBytesNeeded = DWORD(sizeof(lpBuffer))
    _QueryServiceStatusEx(hService, InfoLevel, byref(lpBuffer), sizeof(lpBuffer), byref(cbBytesNeeded))
    return ServiceStatusProcess(lpBuffer)