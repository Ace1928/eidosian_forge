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
class BaseSite(ABC):
    __slots__ = ('_runner', '_ssl_context', '_backlog', '_server')

    def __init__(self, runner: 'BaseRunner', *, shutdown_timeout: float=60.0, ssl_context: Optional[SSLContext]=None, backlog: int=128) -> None:
        if runner.server is None:
            raise RuntimeError('Call runner.setup() before making a site')
        if shutdown_timeout != 60.0:
            msg = 'shutdown_timeout should be set on BaseRunner'
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            runner._shutdown_timeout = shutdown_timeout
        self._runner = runner
        self._ssl_context = ssl_context
        self._backlog = backlog
        self._server: Optional[asyncio.AbstractServer] = None

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    async def start(self) -> None:
        self._runner._reg_site(self)

    async def stop(self) -> None:
        self._runner._check_site(self)
        if self._server is not None:
            self._server.close()
        self._runner._unreg_site(self)