from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def PathAddBackslashW(lpszPath):
    _PathAddBackslashW = windll.shlwapi.PathAddBackslashW
    _PathAddBackslashW.argtypes = [LPWSTR]
    _PathAddBackslashW.restype = LPWSTR
    lpszPath = ctypes.create_unicode_buffer(lpszPath, MAX_PATH)
    retval = _PathAddBackslashW(lpszPath)
    if retval == NULL:
        raise ctypes.WinError()
    return lpszPath.value