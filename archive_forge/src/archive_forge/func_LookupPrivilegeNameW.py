from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def LookupPrivilegeNameW(lpSystemName, lpLuid):
    _LookupPrivilegeNameW = windll.advapi32.LookupPrivilegeNameW
    _LookupPrivilegeNameW.argtypes = [LPWSTR, PLUID, LPWSTR, LPDWORD]
    _LookupPrivilegeNameW.restype = bool
    _LookupPrivilegeNameW.errcheck = RaiseIfZero
    cchName = DWORD(0)
    _LookupPrivilegeNameW(lpSystemName, byref(lpLuid), NULL, byref(cchName))
    lpName = ctypes.create_unicode_buffer(u'', cchName.value)
    _LookupPrivilegeNameW(lpSystemName, byref(lpLuid), byref(lpName), byref(cchName))
    return lpName.value