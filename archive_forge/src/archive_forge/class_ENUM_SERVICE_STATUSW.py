from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
class ENUM_SERVICE_STATUSW(Structure):
    _fields_ = [('lpServiceName', LPWSTR), ('lpDisplayName', LPWSTR), ('ServiceStatus', SERVICE_STATUS)]