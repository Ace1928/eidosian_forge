import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def OpenThread(dwDesiredAccess, bInheritHandle, dwThreadId):
    _OpenThread = windll.kernel32.OpenThread
    _OpenThread.argtypes = [DWORD, BOOL, DWORD]
    _OpenThread.restype = HANDLE
    hThread = _OpenThread(dwDesiredAccess, bool(bInheritHandle), dwThreadId)
    if hThread == NULL:
        raise ctypes.WinError()
    return ThreadHandle(hThread, dwAccess=dwDesiredAccess)