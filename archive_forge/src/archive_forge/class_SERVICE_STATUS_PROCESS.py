from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
class SERVICE_STATUS_PROCESS(Structure):
    _fields_ = SERVICE_STATUS._fields_ + [('dwProcessId', DWORD), ('dwServiceFlags', DWORD)]