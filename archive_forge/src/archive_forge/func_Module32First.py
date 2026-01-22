import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def Module32First(hSnapshot):
    _Module32First = windll.kernel32.Module32First
    _Module32First.argtypes = [HANDLE, LPMODULEENTRY32]
    _Module32First.restype = bool
    me = MODULEENTRY32()
    me.dwSize = sizeof(MODULEENTRY32)
    success = _Module32First(hSnapshot, byref(me))
    if not success:
        if GetLastError() == ERROR_NO_MORE_FILES:
            return None
        raise ctypes.WinError()
    return me