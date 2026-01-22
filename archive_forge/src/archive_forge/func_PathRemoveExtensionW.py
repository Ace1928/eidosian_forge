from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def PathRemoveExtensionW(pszPath):
    _PathRemoveExtensionW = windll.shlwapi.PathRemoveExtensionW
    _PathRemoveExtensionW.argtypes = [LPWSTR]
    pszPath = ctypes.create_unicode_buffer(pszPath, MAX_PATH)
    _PathRemoveExtensionW(pszPath)
    return pszPath.value