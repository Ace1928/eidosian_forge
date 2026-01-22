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
class ConnectionPool:
    """
    Create a connection pool. ``If max_connections`` is set, then this
    object raises :py:class:`~redis.ConnectionError` when the pool's
    limit is reached.

    By default, TCP connections are created unless ``connection_class``
    is specified. Use :py:class:`~redis.UnixDomainSocketConnection` for
    unix sockets.

    Any additional keyword arguments are passed to the constructor of
    ``connection_class``.
    """

    @classmethod
    def from_url(cls: Type[_CP], url: str, **kwargs) -> _CP:
        """
        Return a connection pool configured from the given URL.

        For example::

            redis://[[username]:[password]]@localhost:6379/0
            rediss://[[username]:[password]]@localhost:6379/0
            unix://[username@]/path/to/socket.sock?db=0[&password=password]

        Three URL schemes are supported:

        - `redis://` creates a TCP socket connection. See more at:
          <https://www.iana.org/assignments/uri-schemes/prov/redis>
        - `rediss://` creates a SSL wrapped TCP socket connection. See more at:
          <https://www.iana.org/assignments/uri-schemes/prov/rediss>
        - ``unix://``: creates a Unix Domain Socket connection.

        The username, password, hostname, path and all querystring values
        are passed through urllib.parse.unquote in order to replace any
        percent-encoded values with their corresponding characters.

        There are several ways to specify a database number. The first value
        found will be used:

        1. A ``db`` querystring option, e.g. redis://localhost?db=0

        2. If using the redis:// or rediss:// schemes, the path argument
               of the url, e.g. redis://localhost/0

        3. A ``db`` keyword argument to this function.

        If none of these options are specified, the default db=0 is used.

        All querystring options are cast to their appropriate Python types.
        Boolean arguments can be specified with string values "True"/"False"
        or "Yes"/"No". Values that cannot be properly cast cause a
        ``ValueError`` to be raised. Once parsed, the querystring arguments
        and keyword arguments are passed to the ``ConnectionPool``'s
        class initializer. In the case of conflicting arguments, querystring
        arguments always win.
        """
        url_options = parse_url(url)
        kwargs.update(url_options)
        return cls(**kwargs)

    def __init__(self, connection_class: Type[AbstractConnection]=Connection, max_connections: Optional[int]=None, **connection_kwargs):
        max_connections = max_connections or 2 ** 31
        if not isinstance(max_connections, int) or max_connections < 0:
            raise ValueError('"max_connections" must be a positive integer')
        self.connection_class = connection_class
        self.connection_kwargs = connection_kwargs
        self.max_connections = max_connections
        self._available_connections: List[AbstractConnection] = []
        self._in_use_connections: Set[AbstractConnection] = set()
        self.encoder_class = self.connection_kwargs.get('encoder_class', Encoder)

    def __repr__(self):
        return f'{self.__class__.__name__}<{self.connection_class(**self.connection_kwargs)!r}>'

    def reset(self):
        self._available_connections = []
        self._in_use_connections = weakref.WeakSet()

    def can_get_connection(self) -> bool:
        """Return True if a connection can be retrieved from the pool."""
        return self._available_connections or len(self._in_use_connections) < self.max_connections

    async def get_connection(self, command_name, *keys, **options):
        """Get a connected connection from the pool"""
        connection = self.get_available_connection()
        try:
            await self.ensure_connection(connection)
        except BaseException:
            await self.release(connection)
            raise
        return connection

    def get_available_connection(self):
        """Get a connection from the pool, without making sure it is connected"""
        try:
            connection = self._available_connections.pop()
        except IndexError:
            if len(self._in_use_connections) >= self.max_connections:
                raise ConnectionError('Too many connections') from None
            connection = self.make_connection()
        self._in_use_connections.add(connection)
        return connection

    def get_encoder(self):
        """Return an encoder based on encoding settings"""
        kwargs = self.connection_kwargs
        return self.encoder_class(encoding=kwargs.get('encoding', 'utf-8'), encoding_errors=kwargs.get('encoding_errors', 'strict'), decode_responses=kwargs.get('decode_responses', False))

    def make_connection(self):
        """Create a new connection.  Can be overridden by child classes."""
        return self.connection_class(**self.connection_kwargs)

    async def ensure_connection(self, connection: AbstractConnection):
        """Ensure that the connection object is connected and valid"""
        await connection.connect()
        try:
            if await connection.can_read_destructive():
                raise ConnectionError('Connection has data') from None
        except (ConnectionError, OSError):
            await connection.disconnect()
            await connection.connect()
            if await connection.can_read_destructive():
                raise ConnectionError('Connection not ready') from None

    async def release(self, connection: AbstractConnection):
        """Releases the connection back to the pool"""
        self._in_use_connections.remove(connection)
        self._available_connections.append(connection)

    async def disconnect(self, inuse_connections: bool=True):
        """
        Disconnects connections in the pool

        If ``inuse_connections`` is True, disconnect connections that are
        current in use, potentially by other tasks. Otherwise only disconnect
        connections that are idle in the pool.
        """
        if inuse_connections:
            connections: Iterable[AbstractConnection] = chain(self._available_connections, self._in_use_connections)
        else:
            connections = self._available_connections
        resp = await asyncio.gather(*(connection.disconnect() for connection in connections), return_exceptions=True)
        exc = next((r for r in resp if isinstance(r, BaseException)), None)
        if exc:
            raise exc

    async def aclose(self) -> None:
        """Close the pool, disconnecting all connections"""
        await self.disconnect()

    def set_retry(self, retry: 'Retry') -> None:
        for conn in self._available_connections:
            conn.retry = retry
        for conn in self._in_use_connections:
            conn.retry = retry