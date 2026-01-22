import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def SearchPathW(lpPath, lpFileName, lpExtension):
    _SearchPathW = windll.kernel32.SearchPathW
    _SearchPathW.argtypes = [LPWSTR, LPWSTR, LPWSTR, DWORD, LPWSTR, POINTER(LPWSTR)]
    _SearchPathW.restype = DWORD
    _SearchPathW.errcheck = RaiseIfZero
    if not lpPath:
        lpPath = None
    if not lpExtension:
        lpExtension = None
    nBufferLength = _SearchPathW(lpPath, lpFileName, lpExtension, 0, None, None)
    lpBuffer = ctypes.create_unicode_buffer(u'', nBufferLength + 1)
    lpFilePart = LPWSTR()
    _SearchPathW(lpPath, lpFileName, lpExtension, nBufferLength, lpBuffer, byref(lpFilePart))
    lpFilePart = lpFilePart.value
    lpBuffer = lpBuffer.value
    if lpBuffer == u'':
        if GetLastError() == ERROR_SUCCESS:
            raise ctypes.WinError(ERROR_FILE_NOT_FOUND)
        raise ctypes.WinError()
    return (lpBuffer, lpFilePart)