from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def PathCombineA(lpszDir, lpszFile):
    _PathCombineA = windll.shlwapi.PathCombineA
    _PathCombineA.argtypes = [LPSTR, LPSTR, LPSTR]
    _PathCombineA.restype = LPSTR
    lpszDest = ctypes.create_string_buffer('', max(MAX_PATH, len(lpszDir) + len(lpszFile) + 1))
    retval = _PathCombineA(lpszDest, lpszDir, lpszFile)
    if retval == NULL:
        return None
    return lpszDest.value