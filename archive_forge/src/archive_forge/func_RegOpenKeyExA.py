from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def RegOpenKeyExA(hKey=HKEY_LOCAL_MACHINE, lpSubKey=None, samDesired=KEY_ALL_ACCESS):
    _RegOpenKeyExA = windll.advapi32.RegOpenKeyExA
    _RegOpenKeyExA.argtypes = [HKEY, LPSTR, DWORD, REGSAM, PHKEY]
    _RegOpenKeyExA.restype = LONG
    _RegOpenKeyExA.errcheck = RaiseIfNotErrorSuccess
    hkResult = HKEY(INVALID_HANDLE_VALUE)
    _RegOpenKeyExA(hKey, lpSubKey, 0, samDesired, byref(hkResult))
    return RegistryKeyHandle(hkResult.value)