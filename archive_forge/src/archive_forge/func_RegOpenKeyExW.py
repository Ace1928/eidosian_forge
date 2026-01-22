from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def RegOpenKeyExW(hKey=HKEY_LOCAL_MACHINE, lpSubKey=None, samDesired=KEY_ALL_ACCESS):
    _RegOpenKeyExW = windll.advapi32.RegOpenKeyExW
    _RegOpenKeyExW.argtypes = [HKEY, LPWSTR, DWORD, REGSAM, PHKEY]
    _RegOpenKeyExW.restype = LONG
    _RegOpenKeyExW.errcheck = RaiseIfNotErrorSuccess
    hkResult = HKEY(INVALID_HANDLE_VALUE)
    _RegOpenKeyExW(hKey, lpSubKey, 0, samDesired, byref(hkResult))
    return RegistryKeyHandle(hkResult.value)