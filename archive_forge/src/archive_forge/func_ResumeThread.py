import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def ResumeThread(hThread):
    _ResumeThread = windll.kernel32.ResumeThread
    _ResumeThread.argtypes = [HANDLE]
    _ResumeThread.restype = DWORD
    previousCount = _ResumeThread(hThread)
    if previousCount == DWORD(-1).value:
        raise ctypes.WinError()
    return previousCount