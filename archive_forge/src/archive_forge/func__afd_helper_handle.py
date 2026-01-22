from __future__ import annotations
import enum
import itertools
import socket
import sys
from contextlib import contextmanager
from typing import (
import attrs
from outcome import Value
from .. import _core
from ._io_common import wake_all
from ._run import _public
from ._windows_cffi import (
def _afd_helper_handle() -> Handle:
    rawname = '\\\\.\\GLOBALROOT\\Device\\Afd\\Trio'.encode('utf-16le') + b'\x00\x00'
    rawname_buf = ffi.from_buffer(rawname)
    handle = kernel32.CreateFileW(ffi.cast('LPCWSTR', rawname_buf), FileFlags.SYNCHRONIZE, FileFlags.FILE_SHARE_READ | FileFlags.FILE_SHARE_WRITE, ffi.NULL, FileFlags.OPEN_EXISTING, FileFlags.FILE_FLAG_OVERLAPPED, ffi.NULL)
    if handle == INVALID_HANDLE_VALUE:
        raise_winerror()
    return handle