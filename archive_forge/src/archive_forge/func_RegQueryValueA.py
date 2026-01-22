from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def RegQueryValueA(hKey, lpSubKey=None):
    _RegQueryValueA = windll.advapi32.RegQueryValueA
    _RegQueryValueA.argtypes = [HKEY, LPSTR, LPVOID, PLONG]
    _RegQueryValueA.restype = LONG
    _RegQueryValueA.errcheck = RaiseIfNotErrorSuccess
    cbValue = LONG(0)
    _RegQueryValueA(hKey, lpSubKey, None, byref(cbValue))
    lpValue = ctypes.create_string_buffer(cbValue.value)
    _RegQueryValueA(hKey, lpSubKey, lpValue, byref(cbValue))
    return lpValue.value