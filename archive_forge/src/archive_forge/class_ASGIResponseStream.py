from __future__ import annotations
import typing
import sniffio
from .._models import Request, Response
from .._types import AsyncByteStream
from .base import AsyncBaseTransport
class ASGIResponseStream(AsyncByteStream):

    def __init__(self, body: list[bytes]) -> None:
        self._body = body

    async def __aiter__(self) -> typing.AsyncIterator[bytes]:
        yield b''.join(self._body)