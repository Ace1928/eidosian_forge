from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import LocalFree
def ShellExecuteExW(lpExecInfo):
    _ShellExecuteExW = windll.shell32.ShellExecuteExW
    _ShellExecuteExW.argtypes = [LPSHELLEXECUTEINFOW]
    _ShellExecuteExW.restype = BOOL
    _ShellExecuteExW.errcheck = RaiseIfZero
    _ShellExecuteExW(byref(lpExecInfo))