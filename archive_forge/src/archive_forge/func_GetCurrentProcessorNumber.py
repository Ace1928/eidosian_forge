import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def GetCurrentProcessorNumber():
    _GetCurrentProcessorNumber = windll.kernel32.GetCurrentProcessorNumber
    _GetCurrentProcessorNumber.argtypes = []
    _GetCurrentProcessorNumber.restype = DWORD
    _GetCurrentProcessorNumber.errcheck = RaiseIfZero
    return _GetCurrentProcessorNumber()