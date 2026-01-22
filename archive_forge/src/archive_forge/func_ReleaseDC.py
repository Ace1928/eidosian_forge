from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import GetLastError, SetLastError
def ReleaseDC(hWnd, hDC):
    _ReleaseDC = windll.gdi32.ReleaseDC
    _ReleaseDC.argtypes = [HWND, HDC]
    _ReleaseDC.restype = ctypes.c_int
    _ReleaseDC.errcheck = RaiseIfZero
    _ReleaseDC(hWnd, hDC)