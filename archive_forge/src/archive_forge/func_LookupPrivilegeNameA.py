from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def LookupPrivilegeNameA(lpSystemName, lpLuid):
    _LookupPrivilegeNameA = windll.advapi32.LookupPrivilegeNameA
    _LookupPrivilegeNameA.argtypes = [LPSTR, PLUID, LPSTR, LPDWORD]
    _LookupPrivilegeNameA.restype = bool
    _LookupPrivilegeNameA.errcheck = RaiseIfZero
    cchName = DWORD(0)
    _LookupPrivilegeNameA(lpSystemName, byref(lpLuid), NULL, byref(cchName))
    lpName = ctypes.create_string_buffer('', cchName.value)
    _LookupPrivilegeNameA(lpSystemName, byref(lpLuid), byref(lpName), byref(cchName))
    return lpName.value