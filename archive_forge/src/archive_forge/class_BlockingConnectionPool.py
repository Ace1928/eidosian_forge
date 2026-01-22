import asyncio
import copy
import enum
import inspect
import socket
import ssl
import sys
import warnings
import weakref
from abc import abstractmethod
from itertools import chain
from types import MappingProxyType
from typing import (
from urllib.parse import ParseResult, parse_qs, unquote, urlparse
from redis.asyncio.retry import Retry
from redis.backoff import NoBackoff
from redis.compat import Protocol, TypedDict
from redis.connection import DEFAULT_RESP_VERSION
from redis.credentials import CredentialProvider, UsernamePasswordCredentialProvider
from redis.exceptions import (
from redis.typing import EncodableT
from redis.utils import HIREDIS_AVAILABLE, get_lib_version, str_if_bytes
from .._parsers import (
class BlockingConnectionPool(ConnectionPool):
    """
    A blocking connection pool::

        >>> from redis.asyncio import Redis, BlockingConnectionPool
        >>> client = Redis.from_pool(BlockingConnectionPool())

    It performs the same function as the default
    :py:class:`~redis.asyncio.ConnectionPool` implementation, in that,
    it maintains a pool of reusable connections that can be shared by
    multiple async redis clients.

    The difference is that, in the event that a client tries to get a
    connection from the pool when all of connections are in use, rather than
    raising a :py:class:`~redis.ConnectionError` (as the default
    :py:class:`~redis.asyncio.ConnectionPool` implementation does), it
    blocks the current `Task` for a specified number of seconds until
    a connection becomes available.

    Use ``max_connections`` to increase / decrease the pool size::

        >>> pool = BlockingConnectionPool(max_connections=10)

    Use ``timeout`` to tell it either how many seconds to wait for a connection
    to become available, or to block forever:

        >>> # Block forever.
        >>> pool = BlockingConnectionPool(timeout=None)

        >>> # Raise a ``ConnectionError`` after five seconds if a connection is
        >>> # not available.
        >>> pool = BlockingConnectionPool(timeout=5)
    """

    def __init__(self, max_connections: int=50, timeout: Optional[int]=20, connection_class: Type[AbstractConnection]=Connection, queue_class: Type[asyncio.Queue]=asyncio.LifoQueue, **connection_kwargs):
        super().__init__(connection_class=connection_class, max_connections=max_connections, **connection_kwargs)
        self._condition = asyncio.Condition()
        self.timeout = timeout

    async def get_connection(self, command_name, *keys, **options):
        """Gets a connection from the pool, blocking until one is available"""
        try:
            async with self._condition:
                async with async_timeout(self.timeout):
                    await self._condition.wait_for(self.can_get_connection)
                    connection = super().get_available_connection()
        except asyncio.TimeoutError as err:
            raise ConnectionError('No connection available.') from err
        try:
            await self.ensure_connection(connection)
            return connection
        except BaseException:
            await self.release(connection)
            raise

    async def release(self, connection: AbstractConnection):
        """Releases the connection back to the pool."""
        async with self._condition:
            await super().release(connection)
            self._condition.notify()