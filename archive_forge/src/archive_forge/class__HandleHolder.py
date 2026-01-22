from __future__ import annotations
import sys
from typing import TYPE_CHECKING
from . import _core
from ._abc import ReceiveStream, SendStream
from ._core._windows_cffi import _handle, kernel32, raise_winerror
from ._util import ConflictDetector, final
class _HandleHolder:

    def __init__(self, handle: int) -> None:
        self.handle = -1
        if not isinstance(handle, int):
            raise TypeError('handle must be an int')
        self.handle = handle
        _core.register_with_iocp(self.handle)

    @property
    def closed(self) -> bool:
        return self.handle == -1

    def close(self) -> None:
        if self.closed:
            return
        handle = self.handle
        self.handle = -1
        if not kernel32.CloseHandle(_handle(handle)):
            raise_winerror()

    def __del__(self) -> None:
        self.close()