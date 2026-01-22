from winappdbg.win32.defines import *
def GetProcessImageFileNameW(hProcess):
    _GetProcessImageFileNameW = windll.psapi.GetProcessImageFileNameW
    _GetProcessImageFileNameW.argtypes = [HANDLE, LPWSTR, DWORD]
    _GetProcessImageFileNameW.restype = DWORD
    nSize = MAX_PATH
    while 1:
        lpFilename = ctypes.create_unicode_buffer(u'', nSize)
        nCopied = _GetProcessImageFileNameW(hProcess, lpFilename, nSize)
        if nCopied == 0:
            raise ctypes.WinError()
        if nCopied < nSize - 1:
            break
        nSize = nSize + MAX_PATH
    return lpFilename.value