from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def RegOpenKeyW(hKey=HKEY_LOCAL_MACHINE, lpSubKey=None):
    _RegOpenKeyW = windll.advapi32.RegOpenKeyW
    _RegOpenKeyW.argtypes = [HKEY, LPWSTR, PHKEY]
    _RegOpenKeyW.restype = LONG
    _RegOpenKeyW.errcheck = RaiseIfNotErrorSuccess
    hkResult = HKEY(INVALID_HANDLE_VALUE)
    _RegOpenKeyW(hKey, lpSubKey, byref(hkResult))
    return RegistryKeyHandle(hkResult.value)