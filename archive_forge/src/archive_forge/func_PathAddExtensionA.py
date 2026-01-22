from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def PathAddExtensionA(lpszPath, pszExtension=None):
    _PathAddExtensionA = windll.shlwapi.PathAddExtensionA
    _PathAddExtensionA.argtypes = [LPSTR, LPSTR]
    _PathAddExtensionA.restype = bool
    _PathAddExtensionA.errcheck = RaiseIfZero
    if not pszExtension:
        pszExtension = None
    lpszPath = ctypes.create_string_buffer(lpszPath, MAX_PATH)
    _PathAddExtensionA(lpszPath, pszExtension)
    return lpszPath.value