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
class SockSite(BaseSite):
    __slots__ = ('_sock', '_name')

    def __init__(self, runner: 'BaseRunner', sock: socket.socket, *, shutdown_timeout: float=60.0, ssl_context: Optional[SSLContext]=None, backlog: int=128) -> None:
        super().__init__(runner, shutdown_timeout=shutdown_timeout, ssl_context=ssl_context, backlog=backlog)
        self._sock = sock
        scheme = 'https' if self._ssl_context else 'http'
        if hasattr(socket, 'AF_UNIX') and sock.family == socket.AF_UNIX:
            name = f'{scheme}://unix:{sock.getsockname()}:'
        else:
            host, port = sock.getsockname()[:2]
            name = str(URL.build(scheme=scheme, host=host, port=port))
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def start(self) -> None:
        await super().start()
        loop = asyncio.get_event_loop()
        server = self._runner.server
        assert server is not None
        self._server = await loop.create_server(server, sock=self._sock, ssl=self._ssl_context, backlog=self._backlog)