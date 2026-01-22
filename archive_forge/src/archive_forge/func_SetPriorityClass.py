import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def SetPriorityClass(hProcess, dwPriorityClass=NORMAL_PRIORITY_CLASS):
    _SetPriorityClass = windll.kernel32.SetPriorityClass
    _SetPriorityClass.argtypes = [HANDLE, DWORD]
    _SetPriorityClass.restype = bool
    _SetPriorityClass.errcheck = RaiseIfZero
    _SetPriorityClass(hProcess, dwPriorityClass)