import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def QueryFullProcessImageNameA(hProcess, dwFlags=0):
    _QueryFullProcessImageNameA = windll.kernel32.QueryFullProcessImageNameA
    _QueryFullProcessImageNameA.argtypes = [HANDLE, DWORD, LPSTR, PDWORD]
    _QueryFullProcessImageNameA.restype = bool
    dwSize = MAX_PATH
    while 1:
        lpdwSize = DWORD(dwSize)
        lpExeName = ctypes.create_string_buffer('', lpdwSize.value + 1)
        success = _QueryFullProcessImageNameA(hProcess, dwFlags, lpExeName, byref(lpdwSize))
        if success and 0 < lpdwSize.value < dwSize:
            break
        error = GetLastError()
        if error != ERROR_INSUFFICIENT_BUFFER:
            raise ctypes.WinError(error)
        dwSize = dwSize + 256
        if dwSize > 4096:
            raise ctypes.WinError(error)
    return lpExeName.value