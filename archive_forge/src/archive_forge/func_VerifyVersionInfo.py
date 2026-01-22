from winappdbg.win32.defines import *
def VerifyVersionInfo(lpVersionInfo, dwTypeMask, dwlConditionMask):
    if isinstance(lpVersionInfo, OSVERSIONINFOEXA):
        return VerifyVersionInfoA(lpVersionInfo, dwTypeMask, dwlConditionMask)
    if isinstance(lpVersionInfo, OSVERSIONINFOEXW):
        return VerifyVersionInfoW(lpVersionInfo, dwTypeMask, dwlConditionMask)
    raise TypeError('Bad OSVERSIONINFOEX structure')