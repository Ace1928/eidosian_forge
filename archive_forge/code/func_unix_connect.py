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
def unix_connect(path: Optional[str]=None, uri: str='ws://localhost/', **kwargs: Any) -> Connect:
    """
    Similar to :func:`connect`, but for connecting to a Unix socket.

    This function builds upon the event loop's
    :meth:`~asyncio.loop.create_unix_connection` method.

    It is only available on Unix.

    It's mainly useful for debugging servers listening on Unix sockets.

    Args:
        path: File system path to the Unix socket.
        uri: URI of the WebSocket server; the host is used in the TLS
            handshake for secure connections and in the ``Host`` header.

    """
    return connect(uri=uri, path=path, unix=True, **kwargs)