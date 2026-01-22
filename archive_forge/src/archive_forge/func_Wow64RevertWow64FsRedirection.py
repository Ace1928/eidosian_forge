import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def Wow64RevertWow64FsRedirection(OldValue):
    _Wow64RevertWow64FsRedirection = windll.kernel32.Wow64RevertWow64FsRedirection
    _Wow64RevertWow64FsRedirection.argtypes = [PVOID]
    _Wow64RevertWow64FsRedirection.restype = BOOL
    _Wow64RevertWow64FsRedirection.errcheck = RaiseIfZero
    _Wow64RevertWow64FsRedirection(OldValue)