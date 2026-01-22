from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def PathIsSameRootA(pszPath1, pszPath2):
    _PathIsSameRootA = windll.shlwapi.PathIsSameRootA
    _PathIsSameRootA.argtypes = [LPSTR, LPSTR]
    _PathIsSameRootA.restype = bool
    return _PathIsSameRootA(pszPath1, pszPath2)