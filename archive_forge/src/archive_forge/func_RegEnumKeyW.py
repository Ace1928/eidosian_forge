from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def RegEnumKeyW(hKey, dwIndex):
    _RegEnumKeyW = windll.advapi32.RegEnumKeyW
    _RegEnumKeyW.argtypes = [HKEY, DWORD, LPWSTR, DWORD]
    _RegEnumKeyW.restype = LONG
    cchName = 512
    while True:
        lpName = ctypes.create_unicode_buffer(cchName)
        errcode = _RegEnumKeyW(hKey, dwIndex, lpName, cchName * 2)
        if errcode != ERROR_MORE_DATA:
            break
        cchName = cchName + 512
        if cchName > 32768:
            raise ctypes.WinError(errcode)
    if errcode == ERROR_NO_MORE_ITEMS:
        return None
    if errcode != ERROR_SUCCESS:
        raise ctypes.WinError(errcode)
    return lpName.value