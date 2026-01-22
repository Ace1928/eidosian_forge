from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def PathRemoveArgsA(pszPath):
    _PathRemoveArgsA = windll.shlwapi.PathRemoveArgsA
    _PathRemoveArgsA.argtypes = [LPSTR]
    pszPath = ctypes.create_string_buffer(pszPath, MAX_PATH)
    _PathRemoveArgsA(pszPath)
    return pszPath.value