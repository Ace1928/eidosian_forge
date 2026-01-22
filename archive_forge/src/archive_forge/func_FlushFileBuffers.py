import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def FlushFileBuffers(hFile):
    _FlushFileBuffers = windll.kernel32.FlushFileBuffers
    _FlushFileBuffers.argtypes = [HANDLE]
    _FlushFileBuffers.restype = bool
    _FlushFileBuffers.errcheck = RaiseIfZero
    _FlushFileBuffers(hFile)