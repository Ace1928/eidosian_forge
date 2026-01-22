from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def LookupAccountSidA(lpSystemName, lpSid):
    _LookupAccountSidA = windll.advapi32.LookupAccountSidA
    _LookupAccountSidA.argtypes = [LPSTR, PSID, LPSTR, LPDWORD, LPSTR, LPDWORD, LPDWORD]
    _LookupAccountSidA.restype = bool
    cchName = DWORD(0)
    cchReferencedDomainName = DWORD(0)
    peUse = DWORD(0)
    _LookupAccountSidA(lpSystemName, lpSid, None, byref(cchName), None, byref(cchReferencedDomainName), byref(peUse))
    error = GetLastError()
    if error != ERROR_INSUFFICIENT_BUFFER:
        raise ctypes.WinError(error)
    lpName = ctypes.create_string_buffer('', cchName + 1)
    lpReferencedDomainName = ctypes.create_string_buffer('', cchReferencedDomainName + 1)
    success = _LookupAccountSidA(lpSystemName, lpSid, lpName, byref(cchName), lpReferencedDomainName, byref(cchReferencedDomainName), byref(peUse))
    if not success:
        raise ctypes.WinError()
    return (lpName.value, lpReferencedDomainName.value, peUse.value)