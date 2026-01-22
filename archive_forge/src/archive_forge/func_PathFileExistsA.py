from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def PathFileExistsA(pszPath):
    _PathFileExistsA = windll.shlwapi.PathFileExistsA
    _PathFileExistsA.argtypes = [LPSTR]
    _PathFileExistsA.restype = bool
    return _PathFileExistsA(pszPath)