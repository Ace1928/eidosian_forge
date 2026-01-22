import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def GetCurrentProcessId():
    _GetCurrentProcessId = windll.kernel32.GetCurrentProcessId
    _GetCurrentProcessId.argtypes = []
    _GetCurrentProcessId.restype = DWORD
    return _GetCurrentProcessId()