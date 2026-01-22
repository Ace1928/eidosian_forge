from winappdbg.win32.defines import *
def VerifyVersionInfoA(lpVersionInfo, dwTypeMask, dwlConditionMask):
    _VerifyVersionInfoA = windll.kernel32.VerifyVersionInfoA
    _VerifyVersionInfoA.argtypes = [LPOSVERSIONINFOEXA, DWORD, DWORDLONG]
    _VerifyVersionInfoA.restype = bool
    return _VerifyVersionInfoA(byref(lpVersionInfo), dwTypeMask, dwlConditionMask)