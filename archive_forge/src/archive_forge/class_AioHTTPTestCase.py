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
class AioHTTPTestCase(IsolatedAsyncioTestCase):
    """A base class to allow for unittest web applications using aiohttp.

    Provides the following:

    * self.client (aiohttp.test_utils.TestClient): an aiohttp test client.
    * self.loop (asyncio.BaseEventLoop): the event loop in which the
        application and server are running.
    * self.app (aiohttp.web.Application): the application returned by
        self.get_application()

    Note that the TestClient's methods are asynchronous: you have to
    execute function on the test client using asynchronous methods.
    """

    async def get_application(self) -> Application:
        """Get application.

        This method should be overridden
        to return the aiohttp.web.Application
        object to test.
        """
        return self.get_app()

    def get_app(self) -> Application:
        """Obsolete method used to constructing web application.

        Use .get_application() coroutine instead.
        """
        raise RuntimeError('Did you forget to define get_application()?')

    async def asyncSetUp(self) -> None:
        self.loop = asyncio.get_running_loop()
        return await self.setUpAsync()

    async def setUpAsync(self) -> None:
        self.app = await self.get_application()
        self.server = await self.get_server(self.app)
        self.client = await self.get_client(self.server)
        await self.client.start_server()

    async def asyncTearDown(self) -> None:
        return await self.tearDownAsync()

    async def tearDownAsync(self) -> None:
        await self.client.close()

    async def get_server(self, app: Application) -> TestServer:
        """Return a TestServer instance."""
        return TestServer(app, loop=self.loop)

    async def get_client(self, server: TestServer) -> TestClient:
        """Return a TestClient instance."""
        return TestClient(server, loop=self.loop)