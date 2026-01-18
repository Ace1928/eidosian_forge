from __future__ import annotations
import asyncio
import functools
import logging
import random
import urllib.parse
import warnings
from types import TracebackType
from typing import (
from ..datastructures import Headers, HeadersLike
from ..exceptions import (
from ..extensions import ClientExtensionFactory, Extension
from ..extensions.permessage_deflate import enable_client_permessage_deflate
from ..headers import (
from ..http import USER_AGENT
from ..typing import ExtensionHeader, LoggerLike, Origin, Subprotocol
from ..uri import WebSocketURI, parse_uri
from .compatibility import asyncio_timeout
from .handshake import build_request, check_response
from .http import read_response
from .protocol import WebSocketCommonProtocol
def write_http_request(self, path: str, headers: Headers) -> None:
    """
        Write request line and headers to the HTTP request.

        """
    self.path = path
    self.request_headers = headers
    if self.debug:
        self.logger.debug('> GET %s HTTP/1.1', path)
        for key, value in headers.raw_items():
            self.logger.debug('> %s: %s', key, value)
    request = f'GET {path} HTTP/1.1\r\n'
    request += str(headers)
    self.transport.write(request.encode())