from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import LocalFree
def ShellExecuteExA(lpExecInfo):
    _ShellExecuteExA = windll.shell32.ShellExecuteExA
    _ShellExecuteExA.argtypes = [LPSHELLEXECUTEINFOA]
    _ShellExecuteExA.restype = BOOL
    _ShellExecuteExA.errcheck = RaiseIfZero
    _ShellExecuteExA(byref(lpExecInfo))