from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def PathFindNextComponentA(pszPath):
    _PathFindNextComponentA = windll.shlwapi.PathFindNextComponentA
    _PathFindNextComponentA.argtypes = [LPSTR]
    _PathFindNextComponentA.restype = LPSTR
    pszPath = ctypes.create_string_buffer(pszPath)
    return _PathFindNextComponentA(pszPath)