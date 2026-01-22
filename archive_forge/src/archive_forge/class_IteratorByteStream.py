from __future__ import annotations
import inspect
import warnings
from json import dumps as json_dumps
from typing import (
from urllib.parse import urlencode
from ._exceptions import StreamClosed, StreamConsumed
from ._multipart import MultipartStream
from ._types import (
from ._utils import peek_filelike_length, primitive_value_to_str
class IteratorByteStream(SyncByteStream):
    CHUNK_SIZE = 65536

    def __init__(self, stream: Iterable[bytes]) -> None:
        self._stream = stream
        self._is_stream_consumed = False
        self._is_generator = inspect.isgenerator(stream)

    def __iter__(self) -> Iterator[bytes]:
        if self._is_stream_consumed and self._is_generator:
            raise StreamConsumed()
        self._is_stream_consumed = True
        if hasattr(self._stream, 'read'):
            chunk = self._stream.read(self.CHUNK_SIZE)
            while chunk:
                yield chunk
                chunk = self._stream.read(self.CHUNK_SIZE)
        else:
            for part in self._stream:
                yield part