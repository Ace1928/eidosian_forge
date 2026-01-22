from winappdbg.win32.defines import *
from winappdbg.win32.version import *
from winappdbg.win32.kernel32 import *
def StackWalk64(MachineType, hProcess, hThread, StackFrame, ContextRecord=None, ReadMemoryRoutine=None, FunctionTableAccessRoutine=None, GetModuleBaseRoutine=None, TranslateAddress=None):
    _StackWalk64 = windll.dbghelp.StackWalk64
    _StackWalk64.argtypes = [DWORD, HANDLE, HANDLE, LPSTACKFRAME64, PVOID, PREAD_PROCESS_MEMORY_ROUTINE64, PFUNCTION_TABLE_ACCESS_ROUTINE64, PGET_MODULE_BASE_ROUTINE64, PTRANSLATE_ADDRESS_ROUTINE64]
    _StackWalk64.restype = bool
    pReadMemoryRoutine = None
    if ReadMemoryRoutine:
        pReadMemoryRoutine = PREAD_PROCESS_MEMORY_ROUTINE64(ReadMemoryRoutine)
    else:
        pReadMemoryRoutine = ctypes.cast(None, PREAD_PROCESS_MEMORY_ROUTINE64)
    pFunctionTableAccessRoutine = None
    if FunctionTableAccessRoutine:
        pFunctionTableAccessRoutine = PFUNCTION_TABLE_ACCESS_ROUTINE64(FunctionTableAccessRoutine)
    else:
        pFunctionTableAccessRoutine = ctypes.cast(None, PFUNCTION_TABLE_ACCESS_ROUTINE64)
    pGetModuleBaseRoutine = None
    if GetModuleBaseRoutine:
        pGetModuleBaseRoutine = PGET_MODULE_BASE_ROUTINE64(GetModuleBaseRoutine)
    else:
        pGetModuleBaseRoutine = ctypes.cast(None, PGET_MODULE_BASE_ROUTINE64)
    pTranslateAddress = None
    if TranslateAddress:
        pTranslateAddress = PTRANSLATE_ADDRESS_ROUTINE64(TranslateAddress)
    else:
        pTranslateAddress = ctypes.cast(None, PTRANSLATE_ADDRESS_ROUTINE64)
    pContextRecord = None
    if ContextRecord is None:
        ContextRecord = GetThreadContext(hThread, raw=True)
    pContextRecord = PCONTEXT(ContextRecord)
    ret = _StackWalk64(MachineType, hProcess, hThread, byref(StackFrame), pContextRecord, pReadMemoryRoutine, pFunctionTableAccessRoutine, pGetModuleBaseRoutine, pTranslateAddress)
    return ret