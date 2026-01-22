from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def GetUserNameW():
    _GetUserNameW = windll.advapi32.GetUserNameW
    _GetUserNameW.argtypes = [LPWSTR, LPDWORD]
    _GetUserNameW.restype = bool
    nSize = DWORD(0)
    _GetUserNameW(None, byref(nSize))
    error = GetLastError()
    if error != ERROR_INSUFFICIENT_BUFFER:
        raise ctypes.WinError(error)
    lpBuffer = ctypes.create_unicode_buffer(u'', nSize.value + 1)
    success = _GetUserNameW(lpBuffer, byref(nSize))
    if not success:
        raise ctypes.WinError()
    return lpBuffer.value