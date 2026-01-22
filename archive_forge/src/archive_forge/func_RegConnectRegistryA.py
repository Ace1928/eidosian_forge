from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def RegConnectRegistryA(lpMachineName=None, hKey=HKEY_LOCAL_MACHINE):
    _RegConnectRegistryA = windll.advapi32.RegConnectRegistryA
    _RegConnectRegistryA.argtypes = [LPSTR, HKEY, PHKEY]
    _RegConnectRegistryA.restype = LONG
    _RegConnectRegistryA.errcheck = RaiseIfNotErrorSuccess
    hkResult = HKEY(INVALID_HANDLE_VALUE)
    _RegConnectRegistryA(lpMachineName, hKey, byref(hkResult))
    return RegistryKeyHandle(hkResult.value)