import ctypes
import os
import struct
import subprocess
import sys
import time
from contextlib import contextmanager
import platform
import traceback
import os, time, sys
@contextmanager
def _win_write_to_shared_named_memory(python_code, pid):
    from winappdbg.win32 import defines
    from winappdbg.win32.kernel32 import CreateFileMapping, MapViewOfFile, CloseHandle, UnmapViewOfFile
    memmove = ctypes.cdll.msvcrt.memmove
    memmove.argtypes = [ctypes.c_void_p, ctypes.c_void_p, defines.SIZE_T]
    memmove.restype = ctypes.c_void_p
    BUFSIZE = 2048
    assert isinstance(python_code, bytes)
    assert len(python_code) > 0, 'Python code must not be empty.'
    assert len(python_code) < BUFSIZE - 1, 'Python code must have at most %s bytes (found: %s)' % (BUFSIZE - 1, len(python_code))
    python_code += b'\x00' * (BUFSIZE - len(python_code))
    assert python_code.endswith(b'\x00')
    INVALID_HANDLE_VALUE = -1
    PAGE_READWRITE = 4
    FILE_MAP_WRITE = 2
    filemap = CreateFileMapping(INVALID_HANDLE_VALUE, 0, PAGE_READWRITE, 0, BUFSIZE, u'__pydevd_pid_code_to_run__%s' % (pid,))
    if filemap == INVALID_HANDLE_VALUE or filemap is None:
        raise Exception('Failed to create named file mapping (ctypes: CreateFileMapping): %s' % (filemap,))
    try:
        view = MapViewOfFile(filemap, FILE_MAP_WRITE, 0, 0, 0)
        if not view:
            raise Exception('Failed to create view of named file mapping (ctypes: MapViewOfFile).')
        try:
            memmove(view, python_code, BUFSIZE)
            yield
        finally:
            UnmapViewOfFile(view)
    finally:
        CloseHandle(filemap)