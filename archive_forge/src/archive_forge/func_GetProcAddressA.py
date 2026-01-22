import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def GetProcAddressA(hModule, lpProcName):
    _GetProcAddress = windll.kernel32.GetProcAddress
    _GetProcAddress.argtypes = [HMODULE, LPVOID]
    _GetProcAddress.restype = LPVOID
    if type(lpProcName) in (type(0), type(long(0))):
        lpProcName = LPVOID(lpProcName)
        if lpProcName.value & ~65535:
            raise ValueError('Ordinal number too large: %d' % lpProcName.value)
    elif type(lpProcName) == type(compat.b('')):
        lpProcName = ctypes.c_char_p(lpProcName)
    else:
        raise TypeError(str(type(lpProcName)))
    return _GetProcAddress(hModule, lpProcName)