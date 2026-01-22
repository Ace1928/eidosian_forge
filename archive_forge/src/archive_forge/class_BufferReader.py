from __future__ import annotations
import io
from typing import Callable
from typing_extensions import override
class BufferReader(io.BytesIO):

    def __init__(self, buf: bytes=b'', desc: str | None=None) -> None:
        super().__init__(buf)
        self._len = len(buf)
        self._progress = 0
        self._callback = progress(len(buf), desc=desc)

    def __len__(self) -> int:
        return self._len

    @override
    def read(self, n: int | None=-1) -> bytes:
        chunk = io.BytesIO.read(self, n)
        self._progress += len(chunk)
        try:
            self._callback(self._progress)
        except Exception as e:
            raise CancelledError('The upload was cancelled: {}'.format(e)) from e
        return chunk