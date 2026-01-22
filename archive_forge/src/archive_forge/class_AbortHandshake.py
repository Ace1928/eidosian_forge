from __future__ import annotations
import http
from typing import Optional
from . import datastructures, frames, http11
from .typing import StatusLike
class AbortHandshake(InvalidHandshake):
    """
    Raised to abort the handshake on purpose and return an HTTP response.

    This exception is an implementation detail.

    The public API
    is :meth:`~websockets.server.WebSocketServerProtocol.process_request`.

    Attributes:
        status (~http.HTTPStatus): HTTP status code.
        headers (Headers): HTTP response headers.
        body (bytes): HTTP response body.
    """

    def __init__(self, status: StatusLike, headers: datastructures.HeadersLike, body: bytes=b'') -> None:
        self.status = http.HTTPStatus(status)
        self.headers = datastructures.Headers(headers)
        self.body = body

    def __str__(self) -> str:
        return f'HTTP {self.status:d}, {len(self.headers)} headers, {len(self.body)} bytes'