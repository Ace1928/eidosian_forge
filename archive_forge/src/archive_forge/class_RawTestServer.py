import asyncio
import contextlib
import gc
import inspect
import ipaddress
import os
import socket
import sys
import warnings
from abc import ABC, abstractmethod
from types import TracebackType
from typing import (
from unittest import IsolatedAsyncioTestCase, mock
from aiosignal import Signal
from multidict import CIMultiDict, CIMultiDictProxy
from yarl import URL
import aiohttp
from aiohttp.client import _RequestContextManager, _WSRequestContextManager
from . import ClientSession, hdrs
from .abc import AbstractCookieJar
from .client_reqrep import ClientResponse
from .client_ws import ClientWebSocketResponse
from .helpers import sentinel
from .http import HttpVersion, RawRequestMessage
from .typedefs import StrOrURL
from .web import (
from .web_protocol import _RequestHandler
class RawTestServer(BaseTestServer):

    def __init__(self, handler: _RequestHandler, *, scheme: Union[str, object]=sentinel, host: str='127.0.0.1', port: Optional[int]=None, **kwargs: Any) -> None:
        self._handler = handler
        super().__init__(scheme=scheme, host=host, port=port, **kwargs)

    async def _make_runner(self, debug: bool=True, **kwargs: Any) -> ServerRunner:
        srv = Server(self._handler, loop=self._loop, debug=debug, **kwargs)
        return ServerRunner(srv, debug=debug, **kwargs)