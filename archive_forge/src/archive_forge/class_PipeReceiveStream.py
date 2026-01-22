from __future__ import annotations
import sys
from typing import TYPE_CHECKING
from . import _core
from ._abc import ReceiveStream, SendStream
from ._core._windows_cffi import _handle, kernel32, raise_winerror
from ._util import ConflictDetector, final
@final
class PipeReceiveStream(ReceiveStream):
    """Represents a receive stream over an os.pipe object."""

    def __init__(self, handle: int) -> None:
        self._handle_holder = _HandleHolder(handle)
        self._conflict_detector = ConflictDetector('another task is currently using this pipe')

    async def receive_some(self, max_bytes: int | None=None) -> bytes:
        with self._conflict_detector:
            if self._handle_holder.closed:
                raise _core.ClosedResourceError('this pipe is already closed')
            if max_bytes is None:
                max_bytes = DEFAULT_RECEIVE_SIZE
            else:
                if not isinstance(max_bytes, int):
                    raise TypeError('max_bytes must be integer >= 1')
                if max_bytes < 1:
                    raise ValueError('max_bytes must be integer >= 1')
            buffer = bytearray(max_bytes)
            try:
                size = await _core.readinto_overlapped(self._handle_holder.handle, buffer)
            except BrokenPipeError:
                if self._handle_holder.closed:
                    raise _core.ClosedResourceError('another task closed this pipe') from None
                await _core.checkpoint()
                return b''
            else:
                del buffer[size:]
                return buffer

    def close(self) -> None:
        self._handle_holder.close()

    async def aclose(self) -> None:
        self.close()
        await _core.checkpoint()