from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import LocalFree
def FindExecutableA(lpFile, lpDirectory=None):
    _FindExecutableA = windll.shell32.FindExecutableA
    _FindExecutableA.argtypes = [LPSTR, LPSTR, LPSTR]
    _FindExecutableA.restype = HINSTANCE
    lpResult = ctypes.create_string_buffer(MAX_PATH)
    success = _FindExecutableA(lpFile, lpDirectory, lpResult)
    success = ctypes.cast(success, ctypes.c_void_p)
    success = success.value
    if not success > 32:
        raise ctypes.WinError(success)
    return lpResult.value