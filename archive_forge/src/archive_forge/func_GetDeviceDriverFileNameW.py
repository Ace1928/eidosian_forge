from winappdbg.win32.defines import *
def GetDeviceDriverFileNameW(ImageBase):
    _GetDeviceDriverFileNameW = windll.psapi.GetDeviceDriverFileNameW
    _GetDeviceDriverFileNameW.argtypes = [LPVOID, LPWSTR, DWORD]
    _GetDeviceDriverFileNameW.restype = DWORD
    nSize = MAX_PATH
    while 1:
        lpFilename = ctypes.create_unicode_buffer(u'', nSize)
        nCopied = ctypes.windll.psapi.GetDeviceDriverFileNameW(ImageBase, lpFilename, nSize)
        if nCopied == 0:
            raise ctypes.WinError()
        if nCopied < nSize - 1:
            break
        nSize = nSize + MAX_PATH
    return lpFilename.value