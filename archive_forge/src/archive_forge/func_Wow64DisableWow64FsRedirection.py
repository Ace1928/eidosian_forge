import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def Wow64DisableWow64FsRedirection():
    _Wow64DisableWow64FsRedirection = windll.kernel32.Wow64DisableWow64FsRedirection
    _Wow64DisableWow64FsRedirection.argtypes = [PPVOID]
    _Wow64DisableWow64FsRedirection.restype = BOOL
    _Wow64DisableWow64FsRedirection.errcheck = RaiseIfZero
    OldValue = PVOID(None)
    _Wow64DisableWow64FsRedirection(byref(OldValue))
    return OldValue