from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def GetWindowThreadProcessId(hWnd):
    _GetWindowThreadProcessId = windll.user32.GetWindowThreadProcessId
    _GetWindowThreadProcessId.argtypes = [HWND, LPDWORD]
    _GetWindowThreadProcessId.restype = DWORD
    _GetWindowThreadProcessId.errcheck = RaiseIfZero
    dwProcessId = DWORD(0)
    dwThreadId = _GetWindowThreadProcessId(hWnd, byref(dwProcessId))
    return (dwThreadId, dwProcessId.value)