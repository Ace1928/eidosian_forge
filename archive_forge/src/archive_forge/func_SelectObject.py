from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import GetLastError, SetLastError
def SelectObject(hdc, hgdiobj):
    _SelectObject = windll.gdi32.SelectObject
    _SelectObject.argtypes = [HDC, HGDIOBJ]
    _SelectObject.restype = HGDIOBJ
    _SelectObject.errcheck = RaiseIfZero
    return _SelectObject(hdc, hgdiobj)