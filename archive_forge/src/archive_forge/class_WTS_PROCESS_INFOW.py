from winappdbg.win32.defines import *
from winappdbg.win32.advapi32 import *
class WTS_PROCESS_INFOW(Structure):
    _fields_ = [('SessionId', DWORD), ('ProcessId', DWORD), ('pProcessName', LPWSTR), ('pUserSid', PSID)]