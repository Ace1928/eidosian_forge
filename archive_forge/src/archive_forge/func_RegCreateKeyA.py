from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def RegCreateKeyA(hKey=HKEY_LOCAL_MACHINE, lpSubKey=None):
    _RegCreateKeyA = windll.advapi32.RegCreateKeyA
    _RegCreateKeyA.argtypes = [HKEY, LPSTR, PHKEY]
    _RegCreateKeyA.restype = LONG
    _RegCreateKeyA.errcheck = RaiseIfNotErrorSuccess
    hkResult = HKEY(INVALID_HANDLE_VALUE)
    _RegCreateKeyA(hKey, lpSubKey, byref(hkResult))
    return RegistryKeyHandle(hkResult.value)