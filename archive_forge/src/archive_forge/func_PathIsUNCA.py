from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def PathIsUNCA(pszPath):
    _PathIsUNCA = windll.shlwapi.PathIsUNCA
    _PathIsUNCA.argtypes = [LPSTR]
    _PathIsUNCA.restype = bool
    return _PathIsUNCA(pszPath)