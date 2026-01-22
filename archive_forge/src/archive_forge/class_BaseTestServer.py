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
class BaseTestServer(ABC):
    __test__ = False

    def __init__(self, *, scheme: Union[str, object]=sentinel, loop: Optional[asyncio.AbstractEventLoop]=None, host: str='127.0.0.1', port: Optional[int]=None, skip_url_asserts: bool=False, socket_factory: Callable[[str, int, socket.AddressFamily], socket.socket]=get_port_socket, **kwargs: Any) -> None:
        self._loop = loop
        self.runner: Optional[BaseRunner] = None
        self._root: Optional[URL] = None
        self.host = host
        self.port = port
        self._closed = False
        self.scheme = scheme
        self.skip_url_asserts = skip_url_asserts
        self.socket_factory = socket_factory

    async def start_server(self, loop: Optional[asyncio.AbstractEventLoop]=None, **kwargs: Any) -> None:
        if self.runner:
            return
        self._loop = loop
        self._ssl = kwargs.pop('ssl', None)
        self.runner = await self._make_runner(handler_cancellation=True, **kwargs)
        await self.runner.setup()
        if not self.port:
            self.port = 0
        try:
            version = ipaddress.ip_address(self.host).version
        except ValueError:
            version = 4
        family = socket.AF_INET6 if version == 6 else socket.AF_INET
        _sock = self.socket_factory(self.host, self.port, family)
        self.host, self.port = _sock.getsockname()[:2]
        site = SockSite(self.runner, sock=_sock, ssl_context=self._ssl)
        await site.start()
        server = site._server
        assert server is not None
        sockets = server.sockets
        assert sockets is not None
        self.port = sockets[0].getsockname()[1]
        if self.scheme is sentinel:
            if self._ssl:
                scheme = 'https'
            else:
                scheme = 'http'
            self.scheme = scheme
        self._root = URL(f'{self.scheme}://{self.host}:{self.port}')

    @abstractmethod
    async def _make_runner(self, **kwargs: Any) -> BaseRunner:
        pass

    def make_url(self, path: StrOrURL) -> URL:
        assert self._root is not None
        url = URL(path)
        if not self.skip_url_asserts:
            assert not url.is_absolute()
            return self._root.join(url)
        else:
            return URL(str(self._root) + str(path))

    @property
    def started(self) -> bool:
        return self.runner is not None

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def handler(self) -> Server:
        runner = self.runner
        assert runner is not None
        assert runner.server is not None
        return runner.server

    async def close(self) -> None:
        """Close all fixtures created by the test client.

        After that point, the TestClient is no longer usable.

        This is an idempotent function: running close multiple times
        will not have any additional effects.

        close is also run when the object is garbage collected, and on
        exit when used as a context manager.

        """
        if self.started and (not self.closed):
            assert self.runner is not None
            await self.runner.cleanup()
            self._root = None
            self.port = None
            self._closed = True

    def __enter__(self) -> None:
        raise TypeError('Use async with instead')

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_value: Optional[BaseException], traceback: Optional[TracebackType]) -> None:
        pass

    async def __aenter__(self) -> 'BaseTestServer':
        await self.start_server(loop=self._loop)
        return self

    async def __aexit__(self, exc_type: Optional[Type[BaseException]], exc_value: Optional[BaseException], traceback: Optional[TracebackType]) -> None:
        await self.close()