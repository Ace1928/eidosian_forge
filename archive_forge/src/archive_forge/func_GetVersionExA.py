from winappdbg.win32.defines import *
def GetVersionExA():
    _GetVersionExA = windll.kernel32.GetVersionExA
    _GetVersionExA.argtypes = [POINTER(OSVERSIONINFOEXA)]
    _GetVersionExA.restype = bool
    _GetVersionExA.errcheck = RaiseIfZero
    osi = OSVERSIONINFOEXA()
    osi.dwOSVersionInfoSize = sizeof(osi)
    try:
        _GetVersionExA(byref(osi))
    except WindowsError:
        osi = OSVERSIONINFOA()
        osi.dwOSVersionInfoSize = sizeof(osi)
        _GetVersionExA.argtypes = [POINTER(OSVERSIONINFOA)]
        _GetVersionExA(byref(osi))
    return osi