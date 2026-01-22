import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def SuspendThread(hThread):
    _SuspendThread = windll.kernel32.SuspendThread
    _SuspendThread.argtypes = [HANDLE]
    _SuspendThread.restype = DWORD
    previousCount = _SuspendThread(hThread)
    if previousCount == DWORD(-1).value:
        raise ctypes.WinError()
    return previousCount