from winappdbg.win32.defines import *
def GetModuleFileNameExA(hProcess, hModule=None):
    _GetModuleFileNameExA = ctypes.windll.psapi.GetModuleFileNameExA
    _GetModuleFileNameExA.argtypes = [HANDLE, HMODULE, LPSTR, DWORD]
    _GetModuleFileNameExA.restype = DWORD
    nSize = MAX_PATH
    while 1:
        lpFilename = ctypes.create_string_buffer('', nSize)
        nCopied = _GetModuleFileNameExA(hProcess, hModule, lpFilename, nSize)
        if nCopied == 0:
            raise ctypes.WinError()
        if nCopied < nSize - 1:
            break
        nSize = nSize + MAX_PATH
    return lpFilename.value