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
class AppRunner(BaseRunner):
    """Web Application runner"""
    __slots__ = ('_app',)

    def __init__(self, app: Application, *, handle_signals: bool=False, **kwargs: Any) -> None:
        super().__init__(handle_signals=handle_signals, **kwargs)
        if not isinstance(app, Application):
            raise TypeError('The first argument should be web.Application instance, got {!r}'.format(app))
        self._app = app

    @property
    def app(self) -> Application:
        return self._app

    async def shutdown(self) -> None:
        await self._app.shutdown()

    async def _make_server(self) -> Server:
        loop = asyncio.get_event_loop()
        self._app._set_loop(loop)
        self._app.on_startup.freeze()
        await self._app.startup()
        self._app.freeze()
        return self._app._make_handler(loop=loop, **self._kwargs)

    async def _cleanup_server(self) -> None:
        await self._app.cleanup()