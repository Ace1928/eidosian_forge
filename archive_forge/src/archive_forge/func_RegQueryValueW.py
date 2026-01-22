from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def RegQueryValueW(hKey, lpSubKey=None):
    _RegQueryValueW = windll.advapi32.RegQueryValueW
    _RegQueryValueW.argtypes = [HKEY, LPWSTR, LPVOID, PLONG]
    _RegQueryValueW.restype = LONG
    _RegQueryValueW.errcheck = RaiseIfNotErrorSuccess
    cbValue = LONG(0)
    _RegQueryValueW(hKey, lpSubKey, None, byref(cbValue))
    lpValue = ctypes.create_unicode_buffer(cbValue.value * sizeof(WCHAR))
    _RegQueryValueW(hKey, lpSubKey, lpValue, byref(cbValue))
    return lpValue.value