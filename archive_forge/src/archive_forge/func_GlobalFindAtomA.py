import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def GlobalFindAtomA(lpString):
    _GlobalFindAtomA = windll.kernel32.GlobalFindAtomA
    _GlobalFindAtomA.argtypes = [LPSTR]
    _GlobalFindAtomA.restype = ATOM
    _GlobalFindAtomA.errcheck = RaiseIfZero
    return _GlobalFindAtomA(lpString)