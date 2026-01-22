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
class ResponseStream(SyncByteStream):

    def __init__(self, httpcore_stream: typing.Iterable[bytes]) -> None:
        self._httpcore_stream = httpcore_stream

    def __iter__(self) -> typing.Iterator[bytes]:
        with map_httpcore_exceptions():
            for part in self._httpcore_stream:
                yield part

    def close(self) -> None:
        if hasattr(self._httpcore_stream, 'close'):
            self._httpcore_stream.close()