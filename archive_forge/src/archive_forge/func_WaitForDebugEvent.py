import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def WaitForDebugEvent(dwMilliseconds=INFINITE):
    _WaitForDebugEvent = windll.kernel32.WaitForDebugEvent
    _WaitForDebugEvent.argtypes = [LPDEBUG_EVENT, DWORD]
    _WaitForDebugEvent.restype = DWORD
    if not dwMilliseconds and dwMilliseconds != 0:
        dwMilliseconds = INFINITE
    lpDebugEvent = DEBUG_EVENT()
    lpDebugEvent.dwDebugEventCode = 0
    lpDebugEvent.dwProcessId = 0
    lpDebugEvent.dwThreadId = 0
    if dwMilliseconds != INFINITE:
        success = _WaitForDebugEvent(byref(lpDebugEvent), dwMilliseconds)
        if success == 0:
            raise ctypes.WinError()
    else:
        while 1:
            success = _WaitForDebugEvent(byref(lpDebugEvent), 100)
            if success != 0:
                break
            code = GetLastError()
            if code not in (ERROR_SEM_TIMEOUT, WAIT_TIMEOUT):
                raise ctypes.WinError(code)
    return lpDebugEvent