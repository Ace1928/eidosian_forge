import os, sys, threading
import ctypes, msvcrt
from ctypes import POINTER
from ctypes.wintypes import HANDLE, HLOCAL, LPVOID, WORD, DWORD, BOOL, \
def _stdin_raw_nonblock(self):
    """Use the raw Win32 handle of sys.stdin to do non-blocking reads"""
    handle = msvcrt.get_osfhandle(sys.stdin.fileno())
    result = WaitForSingleObject(handle, 100)
    if result == WAIT_FAILED:
        raise ctypes.WinError()
    elif result == WAIT_TIMEOUT:
        print('.', end='')
        return None
    else:
        data = ctypes.create_string_buffer(256)
        bytesRead = DWORD(0)
        print('?', end='')
        if not ReadFile(handle, data, 256, ctypes.byref(bytesRead), None):
            raise ctypes.WinError()
        FlushConsoleInputBuffer(handle)
        data = data.value
        data = data.replace('\r\n', '\n')
        data = data.replace('\r', '\n')
        print(repr(data) + ' ', end='')
        return data