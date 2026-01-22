from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def SetPropW(hWnd, lpString, hData):
    _SetPropW = windll.user32.SetPropW
    _SetPropW.argtypes = [HWND, LPWSTR, HANDLE]
    _SetPropW.restype = BOOL
    _SetPropW.errcheck = RaiseIfZero
    _SetPropW(hWnd, lpString, hData)