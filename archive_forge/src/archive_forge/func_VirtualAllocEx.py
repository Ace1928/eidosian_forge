import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def VirtualAllocEx(hProcess, lpAddress=0, dwSize=4096, flAllocationType=MEM_COMMIT | MEM_RESERVE, flProtect=PAGE_EXECUTE_READWRITE):
    _VirtualAllocEx = windll.kernel32.VirtualAllocEx
    _VirtualAllocEx.argtypes = [HANDLE, LPVOID, SIZE_T, DWORD, DWORD]
    _VirtualAllocEx.restype = LPVOID
    lpAddress = _VirtualAllocEx(hProcess, lpAddress, dwSize, flAllocationType, flProtect)
    if lpAddress == NULL:
        raise ctypes.WinError()
    return lpAddress