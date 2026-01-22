import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def GetTempFileNameW(lpPathName=None, lpPrefixString=u'TMP', uUnique=0):
    _GetTempFileNameW = windll.kernel32.GetTempFileNameW
    _GetTempFileNameW.argtypes = [LPWSTR, LPWSTR, UINT, LPWSTR]
    _GetTempFileNameW.restype = UINT
    if lpPathName is None:
        lpPathName = GetTempPathW()
    lpTempFileName = ctypes.create_unicode_buffer(u'', MAX_PATH)
    uUnique = _GetTempFileNameW(lpPathName, lpPrefixString, uUnique, lpTempFileName)
    if uUnique == 0:
        raise ctypes.WinError()
    return (lpTempFileName.value, uUnique)