from winappdbg.win32.defines import *
def GetModuleInformation(hProcess, hModule, lpmodinfo=None):
    _GetModuleInformation = windll.psapi.GetModuleInformation
    _GetModuleInformation.argtypes = [HANDLE, HMODULE, LPMODULEINFO, DWORD]
    _GetModuleInformation.restype = bool
    _GetModuleInformation.errcheck = RaiseIfZero
    if lpmodinfo is None:
        lpmodinfo = MODULEINFO()
    _GetModuleInformation(hProcess, hModule, byref(lpmodinfo), sizeof(lpmodinfo))
    return lpmodinfo