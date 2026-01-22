import asyncio
import signal
import socket
import warnings
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, List, Optional, Set
from yarl import URL
from .typedefs import PathLike
from .web_app import Application
from .web_server import Server
class TCPSite(BaseSite):
    __slots__ = ('_host', '_port', '_reuse_address', '_reuse_port')

    def __init__(self, runner: 'BaseRunner', host: Optional[str]=None, port: Optional[int]=None, *, shutdown_timeout: float=60.0, ssl_context: Optional[SSLContext]=None, backlog: int=128, reuse_address: Optional[bool]=None, reuse_port: Optional[bool]=None) -> None:
        super().__init__(runner, shutdown_timeout=shutdown_timeout, ssl_context=ssl_context, backlog=backlog)
        self._host = host
        if port is None:
            port = 8443 if self._ssl_context else 8080
        self._port = port
        self._reuse_address = reuse_address
        self._reuse_port = reuse_port

    @property
    def name(self) -> str:
        scheme = 'https' if self._ssl_context else 'http'
        host = '0.0.0.0' if self._host is None else self._host
        return str(URL.build(scheme=scheme, host=host, port=self._port))

    async def start(self) -> None:
        await super().start()
        loop = asyncio.get_event_loop()
        server = self._runner.server
        assert server is not None
        self._server = await loop.create_server(server, self._host, self._port, ssl=self._ssl_context, backlog=self._backlog, reuse_address=self._reuse_address, reuse_port=self._reuse_port)