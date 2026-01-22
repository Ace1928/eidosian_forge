from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def QueryServiceStatus(hService):
    _QueryServiceStatus = windll.advapi32.QueryServiceStatus
    _QueryServiceStatus.argtypes = [SC_HANDLE, LPSERVICE_STATUS]
    _QueryServiceStatus.restype = bool
    _QueryServiceStatus.errcheck = RaiseIfZero
    rawServiceStatus = SERVICE_STATUS()
    _QueryServiceStatus(hService, byref(rawServiceStatus))
    return ServiceStatus(rawServiceStatus)