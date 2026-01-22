from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import LocalFree
def ShellExecuteW(hwnd=None, lpOperation=None, lpFile=None, lpParameters=None, lpDirectory=None, nShowCmd=None):
    _ShellExecuteW = windll.shell32.ShellExecuteW
    _ShellExecuteW.argtypes = [HWND, LPWSTR, LPWSTR, LPWSTR, LPWSTR, INT]
    _ShellExecuteW.restype = HINSTANCE
    if not nShowCmd:
        nShowCmd = 0
    success = _ShellExecuteW(hwnd, lpOperation, lpFile, lpParameters, lpDirectory, nShowCmd)
    success = ctypes.cast(success, c_int)
    success = success.value
    if not success > 32:
        raise ctypes.WinError(success)