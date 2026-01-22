import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def Module32Next(hSnapshot, me=None):
    _Module32Next = windll.kernel32.Module32Next
    _Module32Next.argtypes = [HANDLE, LPMODULEENTRY32]
    _Module32Next.restype = bool
    if me is None:
        me = MODULEENTRY32()
    me.dwSize = sizeof(MODULEENTRY32)
    success = _Module32Next(hSnapshot, byref(me))
    if not success:
        if GetLastError() == ERROR_NO_MORE_FILES:
            return None
        raise ctypes.WinError()
    return me