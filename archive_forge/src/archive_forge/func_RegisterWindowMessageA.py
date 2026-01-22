from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def RegisterWindowMessageA(lpString):
    _RegisterWindowMessageA = windll.user32.RegisterWindowMessageA
    _RegisterWindowMessageA.argtypes = [LPSTR]
    _RegisterWindowMessageA.restype = UINT
    _RegisterWindowMessageA.errcheck = RaiseIfZero
    return _RegisterWindowMessageA(lpString)