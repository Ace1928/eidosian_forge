import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def WaitForMultipleObjects(handles, bWaitAll=False, dwMilliseconds=INFINITE):
    _WaitForMultipleObjects = windll.kernel32.WaitForMultipleObjects
    _WaitForMultipleObjects.argtypes = [DWORD, POINTER(HANDLE), BOOL, DWORD]
    _WaitForMultipleObjects.restype = DWORD
    if not dwMilliseconds and dwMilliseconds != 0:
        dwMilliseconds = INFINITE
    nCount = len(handles)
    lpHandlesType = HANDLE * nCount
    lpHandles = lpHandlesType(*handles)
    if dwMilliseconds != INFINITE:
        r = _WaitForMultipleObjects(byref(lpHandles), bool(bWaitAll), dwMilliseconds)
        if r == WAIT_FAILED:
            raise ctypes.WinError()
    else:
        while 1:
            r = _WaitForMultipleObjects(byref(lpHandles), bool(bWaitAll), 100)
            if r == WAIT_FAILED:
                raise ctypes.WinError()
            if r != WAIT_TIMEOUT:
                break
    return r