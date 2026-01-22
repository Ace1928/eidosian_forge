from winappdbg.win32.defines import *
def EnumProcessModules(hProcess):
    _EnumProcessModules = windll.psapi.EnumProcessModules
    _EnumProcessModules.argtypes = [HANDLE, LPVOID, DWORD, LPDWORD]
    _EnumProcessModules.restype = bool
    _EnumProcessModules.errcheck = RaiseIfZero
    size = 4096
    lpcbNeeded = DWORD(size)
    unit = sizeof(HMODULE)
    while 1:
        lphModule = (HMODULE * (size // unit))()
        _EnumProcessModules(hProcess, byref(lphModule), lpcbNeeded, byref(lpcbNeeded))
        needed = lpcbNeeded.value
        if needed <= size:
            break
        size = needed
    return [lphModule[index] for index in compat.xrange(0, int(needed // unit))]