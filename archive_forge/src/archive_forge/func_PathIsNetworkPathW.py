from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def PathIsNetworkPathW(pszPath):
    _PathIsNetworkPathW = windll.shlwapi.PathIsNetworkPathW
    _PathIsNetworkPathW.argtypes = [LPWSTR]
    _PathIsNetworkPathW.restype = bool
    return _PathIsNetworkPathW(pszPath)