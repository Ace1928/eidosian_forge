from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def ConvertStringSidToSidW(StringSid):
    _ConvertStringSidToSidW = windll.advapi32.ConvertStringSidToSidW
    _ConvertStringSidToSidW.argtypes = [LPWSTR, PVOID]
    _ConvertStringSidToSidW.restype = bool
    _ConvertStringSidToSidW.errcheck = RaiseIfZero
    Sid = PVOID()
    _ConvertStringSidToSidW(StringSid, ctypes.pointer(Sid))
    return Sid.value