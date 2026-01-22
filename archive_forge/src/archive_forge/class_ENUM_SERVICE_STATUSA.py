from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
class ENUM_SERVICE_STATUSA(Structure):
    _fields_ = [('lpServiceName', LPSTR), ('lpDisplayName', LPSTR), ('ServiceStatus', SERVICE_STATUS)]