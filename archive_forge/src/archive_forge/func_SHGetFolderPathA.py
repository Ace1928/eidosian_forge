from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import LocalFree
def SHGetFolderPathA(nFolder, hToken=None, dwFlags=SHGFP_TYPE_CURRENT):
    _SHGetFolderPathA = windll.shell32.SHGetFolderPathA
    _SHGetFolderPathA.argtypes = [HWND, ctypes.c_int, HANDLE, DWORD, LPSTR]
    _SHGetFolderPathA.restype = HRESULT
    _SHGetFolderPathA.errcheck = RaiseIfNotZero
    pszPath = ctypes.create_string_buffer(MAX_PATH + 1)
    _SHGetFolderPathA(None, nFolder, hToken, dwFlags, pszPath)
    return pszPath.value