from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def RegSetValueEx(hKey, lpValueName=None, lpData=None, dwType=None):
    if lpValueName is None:
        if isinstance(lpData, GuessStringType.t_ansi):
            ansi = True
        elif isinstance(lpData, GuessStringType.t_unicode):
            ansi = False
        else:
            ansi = GuessStringType.t_ansi == GuessStringType.t_default
    elif isinstance(lpValueName, GuessStringType.t_ansi):
        ansi = True
    elif isinstance(lpValueName, GuessStringType.t_unicode):
        ansi = False
    else:
        raise TypeError('String expected, got %s instead' % type(lpValueName))
    if dwType is None:
        if lpValueName is None:
            dwType = REG_SZ
        elif lpData is None:
            dwType = REG_NONE
        elif isinstance(lpData, GuessStringType.t_ansi):
            dwType = REG_SZ
        elif isinstance(lpData, GuessStringType.t_unicode):
            dwType = REG_SZ
        elif isinstance(lpData, int):
            dwType = REG_DWORD
        elif isinstance(lpData, long):
            dwType = REG_QWORD
        else:
            dwType = REG_BINARY
    if ansi:
        _RegSetValueEx = windll.advapi32.RegSetValueExA
        _RegSetValueEx.argtypes = [HKEY, LPSTR, DWORD, DWORD, LPVOID, DWORD]
    else:
        _RegSetValueEx = windll.advapi32.RegSetValueExW
        _RegSetValueEx.argtypes = [HKEY, LPWSTR, DWORD, DWORD, LPVOID, DWORD]
    _RegSetValueEx.restype = LONG
    _RegSetValueEx.errcheck = RaiseIfNotErrorSuccess
    if lpData is None:
        DataRef = None
        DataSize = 0
    else:
        if dwType in (REG_DWORD, REG_DWORD_BIG_ENDIAN):
            Data = DWORD(lpData)
        elif dwType == REG_QWORD:
            Data = QWORD(lpData)
        elif dwType in (REG_SZ, REG_EXPAND_SZ):
            if ansi:
                Data = ctypes.create_string_buffer(lpData)
            else:
                Data = ctypes.create_unicode_buffer(lpData)
        elif dwType == REG_MULTI_SZ:
            if ansi:
                Data = ctypes.create_string_buffer('\x00'.join(lpData) + '\x00\x00')
            else:
                Data = ctypes.create_unicode_buffer(u'\x00'.join(lpData) + u'\x00\x00')
        elif dwType == REG_LINK:
            Data = ctypes.create_unicode_buffer(lpData)
        else:
            Data = ctypes.create_string_buffer(lpData)
        DataRef = byref(Data)
        DataSize = sizeof(Data)
    _RegSetValueEx(hKey, lpValueName, 0, dwType, DataRef, DataSize)