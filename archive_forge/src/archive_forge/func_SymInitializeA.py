from winappdbg.win32.defines import *
from winappdbg.win32.version import *
from winappdbg.win32.kernel32 import *
def SymInitializeA(hProcess, UserSearchPath=None, fInvadeProcess=False):
    _SymInitialize = windll.dbghelp.SymInitialize
    _SymInitialize.argtypes = [HANDLE, LPSTR, BOOL]
    _SymInitialize.restype = bool
    _SymInitialize.errcheck = RaiseIfZero
    if not UserSearchPath:
        UserSearchPath = None
    _SymInitialize(hProcess, UserSearchPath, fInvadeProcess)