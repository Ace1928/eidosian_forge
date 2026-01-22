import uuid
import logging
import asyncio
import copy
import enum
import errno
import inspect
import io
import os
import socket
import ssl
import threading
import weakref
from itertools import chain
from types import MappingProxyType
from typing import (
from urllib.parse import ParseResult, parse_qs, unquote, urlparse
import async_timeout
from aiokeydb.v1.backoff import NoBackoff
from aiokeydb.v1.asyncio.retry import Retry
from aiokeydb.v1.compat import Protocol, TypedDict
from aiokeydb.v1.exceptions import (
from aiokeydb.v1.credentials import CredentialProvider, UsernamePasswordCredentialProvider
from aiokeydb.v1.typing import EncodableT, EncodedT
from aiokeydb.v1.utils import HIREDIS_AVAILABLE, str_if_bytes, set_ulimits
class AsyncBlockingConnectionPool(AsyncConnectionPool):
    """
    Thread-safe blocking connection pool::

        >>> from redis.client import Redis
        >>> client = Redis(connection_pool=BlockingConnectionPool())

    It performs the same function as the default
    :py:class:`~redis.ConnectionPool` implementation, in that,
    it maintains a pool of reusable connections that can be shared by
    multiple redis clients (safely across threads if required).

    The difference is that, in the event that a client tries to get a
    connection from the pool when all of connections are in use, rather than
    raising a :py:class:`~redis.ConnectionError` (as the default
    :py:class:`~redis.ConnectionPool` implementation does), it
    makes the client wait ("blocks") for a specified number of seconds until
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

    def __init__(self, max_connections: Optional[int]=None, timeout: Optional[int]=20, connection_class: Type[AsyncConnection]=AsyncConnection, queue_class: Type[asyncio.Queue]=asyncio.LifoQueue, **connection_kwargs):
        max_connections = max_connections or 50
        self.queue_class = queue_class
        self.timeout = timeout
        self._connections: List[AsyncConnection]
        super().__init__(connection_class=connection_class, max_connections=max_connections, **connection_kwargs)

    def reset(self):
        self.pool = self.queue_class(self.max_connections)
        while True:
            try:
                self.pool.put_nowait(None)
            except asyncio.QueueFull:
                break
        self._connections = []
        self.pid = os.getpid()

    def make_connection(self):
        """Make a fresh connection."""
        connection = self.connection_class(**self.connection_kwargs)
        self._connections.append(connection)
        return connection

    async def get_connection(self, command_name, *keys, **options):
        """
        Get a connection, blocking for ``self.timeout`` until a connection
        is available from the pool.

        If the connection returned is ``None`` then creates a new connection.
        Because we use a last-in first-out queue, the existing connections
        (having been returned to the pool after the initial ``None`` values
        were added) will be returned before ``None`` values. This means we only
        create new connections when we need to, i.e.: the actual number of
        connections will only increase in response to demand.
        """
        self._checkpid()
        connection = None
        try:
            async with async_timeout.timeout(self.timeout):
                connection = await self.pool.get()
        except (asyncio.QueueEmpty, asyncio.TimeoutError) as e:
            if not self._auto_reset_enabled:
                raise ConnectionError('No connection available. Try enabling `auto_reset_enabled`') from e
            logger.warning(f'Resetting Pool: {len(self._connections)}/{self.pool.qsize()}/{self.max_connections} due to error: {e}')
            await self.reset_pool()
        if connection is None:
            connection = self.make_connection()
        logger.debug(f'{command_name} {keys} {options} | Got connection: {connection} | {len(self._connections)}/{self.pool.qsize()}/{self.max_connections} ')
        try:
            await connection.connect()
            if command_name in {'PUBLISH', 'SUBSCRIBE', 'UNSUBSCRIBE'} and self.auto_pubsub:
                connection.encoder = self._pubsub_encoder
            try:
                if await connection.can_read_destructive():
                    raise ConnectionError('Connection has data') from None
            except (ConnectionError, OSError):
                await connection.disconnect()
                await connection.connect()
                if await connection.can_read_destructive():
                    raise ConnectionError('Connection not ready') from None
        except BaseException:
            await self.release(connection)
            raise
        return connection

    async def release(self, connection: AsyncConnection):
        """Releases the connection back to the pool."""
        self._checkpid()
        if not self.owns_connection(connection):
            await connection.disconnect()
            self.pool.put_nowait(None)
            return
        try:
            self.pool.put_nowait(connection)
        except asyncio.QueueFull:
            if self._auto_reset_enabled:
                logger.warning(f'Resetting Pool: {len(self._connections)}/{self.pool.qsize()}/{self.max_connections} due to queue full')
                await self.reset_pool()

    async def disconnect(self, inuse_connections: bool=True, raise_exceptions: bool=True):
        """Disconnects all connections in the pool."""
        self._checkpid()
        async with self._lock:
            resp = await asyncio.gather(*(connection.disconnect() for connection in self._connections), return_exceptions=True)
            exc = next((r for r in resp if isinstance(r, BaseException)), None)
            if exc and raise_exceptions:
                raise exc

    async def reset_pool(self, inuse_connections: bool=True, raise_exceptions: bool=False):
        """
        Resets the connection pool
        """
        await self.disconnect(inuse_connections=inuse_connections, raise_exceptions=raise_exceptions)
        self = self.__class__(max_connections=self.max_connections, connection_class=self.connection_class, auto_pubsub=self.auto_pubsub, pubsub_decode_responses=self.pubsub_decode_responses, auto_reset_enabled=self._auto_reset_enabled, timeout=self.timeout, queue_class=self.queue_class, **self.connection_kwargs)