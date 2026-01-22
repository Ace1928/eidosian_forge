import os, sys, threading
import ctypes, msvcrt
from ctypes import POINTER
from ctypes.wintypes import HANDLE, HLOCAL, LPVOID, WORD, DWORD, BOOL, \
def _stdin_thread(self, handle, hprocess, func, stdout_func):
    exitCode = DWORD()
    bytesWritten = DWORD(0)
    while True:
        data = func()
        if data is None:
            if not GetExitCodeProcess(hprocess, ctypes.byref(exitCode)):
                raise ctypes.WinError()
            if exitCode.value != STILL_ACTIVE:
                return
            if not WriteFile(handle, '', 0, ctypes.byref(bytesWritten), None):
                raise ctypes.WinError()
            continue
        if isinstance(data, unicode):
            data = data.encode('utf_8')
        if not isinstance(data, str):
            raise RuntimeError('internal stdin function string error')
        if len(data) == 0:
            return
        stdout_func(data)
        while len(data) != 0:
            if not WriteFile(handle, data, len(data), ctypes.byref(bytesWritten), None):
                if GetLastError() == ERROR_NO_DATA:
                    return
                raise ctypes.WinError()
            data = data[bytesWritten.value:]