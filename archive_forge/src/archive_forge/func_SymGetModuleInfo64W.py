from winappdbg.win32.defines import *
from winappdbg.win32.version import *
from winappdbg.win32.kernel32 import *
def SymGetModuleInfo64W(hProcess, dwAddr):
    _SymGetModuleInfo64W = windll.dbghelp.SymGetModuleInfo64W
    _SymGetModuleInfo64W.argtypes = [HANDLE, DWORD64, PIMAGEHLP_MODULE64W]
    _SymGetModuleInfo64W.restype = bool
    _SymGetModuleInfo64W.errcheck = RaiseIfZero
    ModuleInfo = IMAGEHLP_MODULE64W()
    ModuleInfo.SizeOfStruct = sizeof(ModuleInfo)
    _SymGetModuleInfo64W(hProcess, dwAddr, byref(ModuleInfo))
    return ModuleInfo