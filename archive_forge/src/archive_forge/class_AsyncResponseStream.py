from __future__ import annotations
import contextlib
import typing
from types import TracebackType
import httpcore
from .._config import DEFAULT_LIMITS, Limits, Proxy, create_ssl_context
from .._exceptions import (
from .._models import Request, Response
from .._types import AsyncByteStream, CertTypes, ProxyTypes, SyncByteStream, VerifyTypes
from .._urls import URL
from .base import AsyncBaseTransport, BaseTransport
class AsyncResponseStream(AsyncByteStream):

    def __init__(self, httpcore_stream: typing.AsyncIterable[bytes]) -> None:
        self._httpcore_stream = httpcore_stream

    async def __aiter__(self) -> typing.AsyncIterator[bytes]:
        with map_httpcore_exceptions():
            async for part in self._httpcore_stream:
                yield part

    async def aclose(self) -> None:
        if hasattr(self._httpcore_stream, 'aclose'):
            await self._httpcore_stream.aclose()