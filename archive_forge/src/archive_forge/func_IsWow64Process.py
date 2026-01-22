from winappdbg.win32.defines import *
def IsWow64Process(hProcess):
    _IsWow64Process = windll.kernel32.IsWow64Process
    _IsWow64Process.argtypes = [HANDLE, PBOOL]
    _IsWow64Process.restype = bool
    _IsWow64Process.errcheck = RaiseIfZero
    Wow64Process = BOOL(FALSE)
    _IsWow64Process(hProcess, byref(Wow64Process))
    return bool(Wow64Process)