from __future__ import annotations
import enum
import re
from typing import TYPE_CHECKING, NewType, NoReturn, Protocol, cast
import cffi
class _Ws2(Protocol):
    """Statically typed version of the ws2_32.dll functions we use."""

    def WSAGetLastError(self) -> int:
        ...

    def WSAIoctl(self, socket: CData, dwIoControlCode: WSAIoctls, lpvInBuffer: AlwaysNull, cbInBuffer: int, lpvOutBuffer: CData, cbOutBuffer: int, lpcbBytesReturned: CData, lpOverlapped: AlwaysNull, lpCompletionRoutine: AlwaysNull, /) -> int:
        ...