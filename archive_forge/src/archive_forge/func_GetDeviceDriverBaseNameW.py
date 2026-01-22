from winappdbg.win32.defines import *
def GetDeviceDriverBaseNameW(ImageBase):
    _GetDeviceDriverBaseNameW = windll.psapi.GetDeviceDriverBaseNameW
    _GetDeviceDriverBaseNameW.argtypes = [LPVOID, LPWSTR, DWORD]
    _GetDeviceDriverBaseNameW.restype = DWORD
    nSize = MAX_PATH
    while 1:
        lpBaseName = ctypes.create_unicode_buffer(u'', nSize)
        nCopied = _GetDeviceDriverBaseNameW(ImageBase, lpBaseName, nSize)
        if nCopied == 0:
            raise ctypes.WinError()
        if nCopied < nSize - 1:
            break
        nSize = nSize + MAX_PATH
    return lpBaseName.value