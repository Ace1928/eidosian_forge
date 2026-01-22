from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def OpenThreadToken(ThreadHandle, DesiredAccess, OpenAsSelf=True):
    _OpenThreadToken = windll.advapi32.OpenThreadToken
    _OpenThreadToken.argtypes = [HANDLE, DWORD, BOOL, PHANDLE]
    _OpenThreadToken.restype = bool
    _OpenThreadToken.errcheck = RaiseIfZero
    NewTokenHandle = HANDLE(INVALID_HANDLE_VALUE)
    _OpenThreadToken(ThreadHandle, DesiredAccess, OpenAsSelf, byref(NewTokenHandle))
    return TokenHandle(NewTokenHandle.value)