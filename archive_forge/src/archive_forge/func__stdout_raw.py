import os, sys, threading
import ctypes, msvcrt
from ctypes import POINTER
from ctypes.wintypes import HANDLE, HLOCAL, LPVOID, WORD, DWORD, BOOL, \
def _stdout_raw(self, s):
    """Writes the string to stdout"""
    print(s, end='', file=sys.stdout)
    sys.stdout.flush()