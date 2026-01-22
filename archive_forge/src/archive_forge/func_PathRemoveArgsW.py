from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def PathRemoveArgsW(pszPath):
    _PathRemoveArgsW = windll.shlwapi.PathRemoveArgsW
    _PathRemoveArgsW.argtypes = [LPWSTR]
    pszPath = ctypes.create_unicode_buffer(pszPath, MAX_PATH)
    _PathRemoveArgsW(pszPath)
    return pszPath.value