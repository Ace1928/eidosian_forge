import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def GetStdHandle(nStdHandle):
    _GetStdHandle = windll.kernel32.GetStdHandle
    _GetStdHandle.argytpes = [DWORD]
    _GetStdHandle.restype = HANDLE
    _GetStdHandle.errcheck = RaiseIfZero
    return Handle(_GetStdHandle(nStdHandle), bOwnership=False)