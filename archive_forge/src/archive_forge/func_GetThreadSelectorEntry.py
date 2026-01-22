from winappdbg.win32.defines import *
from winappdbg.win32.version import ARCH_I386
def GetThreadSelectorEntry(hThread, dwSelector):
    _GetThreadSelectorEntry = windll.kernel32.GetThreadSelectorEntry
    _GetThreadSelectorEntry.argtypes = [HANDLE, DWORD, LPLDT_ENTRY]
    _GetThreadSelectorEntry.restype = bool
    _GetThreadSelectorEntry.errcheck = RaiseIfZero
    ldt = LDT_ENTRY()
    _GetThreadSelectorEntry(hThread, dwSelector, byref(ldt))
    return ldt