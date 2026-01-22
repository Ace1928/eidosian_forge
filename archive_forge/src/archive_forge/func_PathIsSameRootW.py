from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def PathIsSameRootW(pszPath1, pszPath2):
    _PathIsSameRootW = windll.shlwapi.PathIsSameRootW
    _PathIsSameRootW.argtypes = [LPWSTR, LPWSTR]
    _PathIsSameRootW.restype = bool
    return _PathIsSameRootW(pszPath1, pszPath2)