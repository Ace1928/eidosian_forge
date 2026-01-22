from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def RegEnumValueW(hKey, dwIndex, bGetData=True):
    return _internal_RegEnumValue(False, hKey, dwIndex, bGetData)