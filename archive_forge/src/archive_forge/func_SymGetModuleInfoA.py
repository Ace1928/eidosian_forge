from winappdbg.win32.defines import *
from winappdbg.win32.version import *
from winappdbg.win32.kernel32 import *
def SymGetModuleInfoA(hProcess, dwAddr):
    _SymGetModuleInfo = windll.dbghelp.SymGetModuleInfo
    _SymGetModuleInfo.argtypes = [HANDLE, DWORD, PIMAGEHLP_MODULE]
    _SymGetModuleInfo.restype = bool
    _SymGetModuleInfo.errcheck = RaiseIfZero
    ModuleInfo = IMAGEHLP_MODULE()
    ModuleInfo.SizeOfStruct = sizeof(ModuleInfo)
    _SymGetModuleInfo(hProcess, dwAddr, byref(ModuleInfo))
    return ModuleInfo