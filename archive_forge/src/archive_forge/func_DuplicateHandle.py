import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def DuplicateHandle(hSourceHandle, hSourceProcessHandle=None, hTargetProcessHandle=None, dwDesiredAccess=STANDARD_RIGHTS_ALL, bInheritHandle=False, dwOptions=DUPLICATE_SAME_ACCESS):
    _DuplicateHandle = windll.kernel32.DuplicateHandle
    _DuplicateHandle.argtypes = [HANDLE, HANDLE, HANDLE, LPHANDLE, DWORD, BOOL, DWORD]
    _DuplicateHandle.restype = bool
    _DuplicateHandle.errcheck = RaiseIfZero
    if hSourceProcessHandle is None:
        hSourceProcessHandle = GetCurrentProcess()
    if hTargetProcessHandle is None:
        hTargetProcessHandle = hSourceProcessHandle
    lpTargetHandle = HANDLE(INVALID_HANDLE_VALUE)
    _DuplicateHandle(hSourceProcessHandle, hSourceHandle, hTargetProcessHandle, byref(lpTargetHandle), dwDesiredAccess, bool(bInheritHandle), dwOptions)
    if isinstance(hSourceHandle, Handle):
        HandleClass = hSourceHandle.__class__
    else:
        HandleClass = Handle
    if hasattr(hSourceHandle, 'dwAccess'):
        return HandleClass(lpTargetHandle.value, dwAccess=hSourceHandle.dwAccess)
    else:
        return HandleClass(lpTargetHandle.value)