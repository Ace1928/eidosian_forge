import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def CreateEventA(lpMutexAttributes=None, bManualReset=False, bInitialState=False, lpName=None):
    _CreateEventA = windll.kernel32.CreateEventA
    _CreateEventA.argtypes = [LPVOID, BOOL, BOOL, LPSTR]
    _CreateEventA.restype = HANDLE
    _CreateEventA.errcheck = RaiseIfZero
    return Handle(_CreateEventA(lpMutexAttributes, bManualReset, bInitialState, lpName))