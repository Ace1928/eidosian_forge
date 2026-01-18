from __future__ import annotations
import asyncio
import email.utils
import functools
import http
import inspect
import logging
import socket
import warnings
from types import TracebackType
from typing import (
from ..datastructures import Headers, HeadersLike, MultipleValuesError
from ..exceptions import (
from ..extensions import Extension, ServerExtensionFactory
from ..extensions.permessage_deflate import enable_server_permessage_deflate
from ..headers import (
from ..http import USER_AGENT
from ..protocol import State
from ..typing import ExtensionHeader, LoggerLike, Origin, StatusLike, Subprotocol
from .compatibility import asyncio_timeout
from .handshake import build_response, check_request
from .http import read_request
from .protocol import WebSocketCommonProtocol
def write_http_response(self, status: http.HTTPStatus, headers: Headers, body: Optional[bytes]=None) -> None:
    """
        Write status line and headers to the HTTP response.

        This coroutine is also able to write a response body.

        """
    self.response_headers = headers
    if self.debug:
        self.logger.debug('> HTTP/1.1 %d %s', status.value, status.phrase)
        for key, value in headers.raw_items():
            self.logger.debug('> %s: %s', key, value)
        if body is not None:
            self.logger.debug('> [body] (%d bytes)', len(body))
    response = f'HTTP/1.1 {status.value} {status.phrase}\r\n'
    response += str(headers)
    self.transport.write(response.encode())
    if body is not None:
        self.transport.write(body)