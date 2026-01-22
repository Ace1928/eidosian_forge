import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def GetPriorityClass(hProcess):
    _GetPriorityClass = windll.kernel32.GetPriorityClass
    _GetPriorityClass.argtypes = [HANDLE]
    _GetPriorityClass.restype = DWORD
    retval = _GetPriorityClass(hProcess)
    if retval == 0:
        raise ctypes.WinError()
    return retval