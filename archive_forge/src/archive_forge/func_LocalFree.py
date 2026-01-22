import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def LocalFree(hMem):
    _LocalFree = windll.kernel32.LocalFree
    _LocalFree.argtypes = [HLOCAL]
    _LocalFree.restype = HLOCAL
    result = _LocalFree(hMem)
    if result != NULL:
        ctypes.WinError()