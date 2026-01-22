from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def RegQueryValueExA(hKey, lpValueName=None, bGetData=True):
    return _internal_RegQueryValueEx(True, hKey, lpValueName, bGetData)