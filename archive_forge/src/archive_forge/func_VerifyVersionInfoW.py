from winappdbg.win32.defines import *
def VerifyVersionInfoW(lpVersionInfo, dwTypeMask, dwlConditionMask):
    _VerifyVersionInfoW = windll.kernel32.VerifyVersionInfoW
    _VerifyVersionInfoW.argtypes = [LPOSVERSIONINFOEXW, DWORD, DWORDLONG]
    _VerifyVersionInfoW.restype = bool
    return _VerifyVersionInfoW(byref(lpVersionInfo), dwTypeMask, dwlConditionMask)