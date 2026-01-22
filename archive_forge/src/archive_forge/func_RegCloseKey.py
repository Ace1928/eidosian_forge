from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def RegCloseKey(hKey):
    if hasattr(hKey, 'value'):
        value = hKey.value
    else:
        value = hKey
    if value in (HKEY_CLASSES_ROOT, HKEY_CURRENT_USER, HKEY_LOCAL_MACHINE, HKEY_USERS, HKEY_PERFORMANCE_DATA, HKEY_CURRENT_CONFIG):
        return
    _RegCloseKey = windll.advapi32.RegCloseKey
    _RegCloseKey.argtypes = [HKEY]
    _RegCloseKey.restype = LONG
    _RegCloseKey.errcheck = RaiseIfNotErrorSuccess
    _RegCloseKey(hKey)