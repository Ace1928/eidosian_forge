from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def PathCanonicalizeA(lpszSrc):
    _PathCanonicalizeA = windll.shlwapi.PathCanonicalizeA
    _PathCanonicalizeA.argtypes = [LPSTR, LPSTR]
    _PathCanonicalizeA.restype = bool
    _PathCanonicalizeA.errcheck = RaiseIfZero
    lpszDst = ctypes.create_string_buffer('', MAX_PATH)
    _PathCanonicalizeA(lpszDst, lpszSrc)
    return lpszDst.value