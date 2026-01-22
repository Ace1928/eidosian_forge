from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def PathRemoveExtensionA(pszPath):
    _PathRemoveExtensionA = windll.shlwapi.PathRemoveExtensionA
    _PathRemoveExtensionA.argtypes = [LPSTR]
    pszPath = ctypes.create_string_buffer(pszPath, MAX_PATH)
    _PathRemoveExtensionA(pszPath)
    return pszPath.value