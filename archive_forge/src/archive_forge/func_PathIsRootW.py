from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def PathIsRootW(pszPath):
    _PathIsRootW = windll.shlwapi.PathIsRootW
    _PathIsRootW.argtypes = [LPWSTR]
    _PathIsRootW.restype = bool
    return _PathIsRootW(pszPath)