from __future__ import annotations
import enum
import re
from typing import TYPE_CHECKING, NewType, NoReturn, Protocol, cast
import cffi
class _Kernel32(Protocol):
    """Statically typed version of the kernel32.dll functions we use."""

    def CreateIoCompletionPort(self, FileHandle: Handle, ExistingCompletionPort: CData | AlwaysNull, CompletionKey: int, NumberOfConcurrentThreads: int, /) -> Handle:
        ...

    def CreateEventA(self, lpEventAttributes: AlwaysNull, bManualReset: bool, bInitialState: bool, lpName: AlwaysNull, /) -> Handle:
        ...

    def SetFileCompletionNotificationModes(self, handle: Handle, flags: CompletionModes, /) -> int:
        ...

    def PostQueuedCompletionStatus(self, CompletionPort: Handle, dwNumberOfBytesTransferred: int, dwCompletionKey: int, lpOverlapped: CData | AlwaysNull, /) -> bool:
        ...

    def CancelIoEx(self, hFile: Handle, lpOverlapped: CData | AlwaysNull, /) -> bool:
        ...

    def WriteFile(self, hFile: Handle, lpBuffer: CData, nNumberOfBytesToWrite: int, lpNumberOfBytesWritten: AlwaysNull, lpOverlapped: _Overlapped, /) -> bool:
        ...

    def ReadFile(self, hFile: Handle, lpBuffer: CData, nNumberOfBytesToRead: int, lpNumberOfBytesRead: AlwaysNull, lpOverlapped: _Overlapped, /) -> bool:
        ...

    def GetQueuedCompletionStatusEx(self, CompletionPort: Handle, lpCompletionPortEntries: CData, ulCount: int, ulNumEntriesRemoved: CData, dwMilliseconds: int, fAlertable: bool | int, /) -> CData:
        ...

    def CreateFileW(self, lpFileName: CData, dwDesiredAccess: FileFlags, dwShareMode: FileFlags, lpSecurityAttributes: AlwaysNull, dwCreationDisposition: FileFlags, dwFlagsAndAttributes: FileFlags, hTemplateFile: AlwaysNull, /) -> Handle:
        ...

    def WaitForSingleObject(self, hHandle: Handle, dwMilliseconds: int, /) -> CData:
        ...

    def WaitForMultipleObjects(self, nCount: int, lpHandles: HandleArray, bWaitAll: bool, dwMilliseconds: int, /) -> ErrorCodes:
        ...

    def SetEvent(self, handle: Handle, /) -> None:
        ...

    def CloseHandle(self, handle: Handle, /) -> bool:
        ...

    def DeviceIoControl(self, hDevice: Handle, dwIoControlCode: int, lpInBuffer: AlwaysNull, nInBufferSize: int, lpOutBuffer: AlwaysNull, nOutBufferSize: int, lpBytesReturned: AlwaysNull, lpOverlapped: CData, /) -> bool:
        ...