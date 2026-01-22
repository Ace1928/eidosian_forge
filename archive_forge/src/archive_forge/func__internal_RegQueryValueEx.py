from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def _internal_RegQueryValueEx(ansi, hKey, lpValueName=None, bGetData=True):
    _RegQueryValueEx = _caller_RegQueryValueEx(ansi)
    cbData = DWORD(0)
    dwType = DWORD(-1)
    _RegQueryValueEx(hKey, lpValueName, None, byref(dwType), None, byref(cbData))
    Type = dwType.value
    if not bGetData:
        return (cbData.value, Type)
    if Type in (REG_DWORD, REG_DWORD_BIG_ENDIAN):
        if cbData.value != 4:
            raise ValueError('REG_DWORD value of size %d' % cbData.value)
        dwData = DWORD(0)
        _RegQueryValueEx(hKey, lpValueName, None, None, byref(dwData), byref(cbData))
        return (dwData.value, Type)
    if Type == REG_QWORD:
        if cbData.value != 8:
            raise ValueError('REG_QWORD value of size %d' % cbData.value)
        qwData = QWORD(long(0))
        _RegQueryValueEx(hKey, lpValueName, None, None, byref(qwData), byref(cbData))
        return (qwData.value, Type)
    if Type in (REG_SZ, REG_EXPAND_SZ):
        if ansi:
            szData = ctypes.create_string_buffer(cbData.value)
        else:
            szData = ctypes.create_unicode_buffer(cbData.value)
        _RegQueryValueEx(hKey, lpValueName, None, None, byref(szData), byref(cbData))
        return (szData.value, Type)
    if Type == REG_MULTI_SZ:
        if ansi:
            szData = ctypes.create_string_buffer(cbData.value)
        else:
            szData = ctypes.create_unicode_buffer(cbData.value)
        _RegQueryValueEx(hKey, lpValueName, None, None, byref(szData), byref(cbData))
        Data = szData[:]
        if ansi:
            aData = Data.split('\x00')
        else:
            aData = Data.split(u'\x00')
        aData = [token for token in aData if token]
        return (aData, Type)
    if Type == REG_LINK:
        szData = ctypes.create_unicode_buffer(cbData.value)
        _RegQueryValueEx(hKey, lpValueName, None, None, byref(szData), byref(cbData))
        return (szData.value, Type)
    szData = ctypes.create_string_buffer(cbData.value)
    _RegQueryValueEx(hKey, lpValueName, None, None, byref(szData), byref(cbData))
    return (szData.raw, Type)