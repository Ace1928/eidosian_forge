import asyncio
import collections
import random
import socket
import ssl
import warnings
from typing import (
from redis._parsers import AsyncCommandsParser, Encoder
from redis._parsers.helpers import (
from redis.asyncio.client import ResponseCallbackT
from redis.asyncio.connection import Connection, DefaultParser, SSLConnection, parse_url
from redis.asyncio.lock import Lock
from redis.asyncio.retry import Retry
from redis.backoff import default_backoff
from redis.client import EMPTY_RESPONSE, NEVER_DECODE, AbstractRedis
from redis.cluster import (
from redis.commands import READ_COMMANDS, AsyncRedisClusterCommands
from redis.crc import REDIS_CLUSTER_HASH_SLOTS, key_slot
from redis.credentials import CredentialProvider
from redis.exceptions import (
from redis.typing import AnyKeyT, EncodableT, KeyT
from redis.utils import (
class ClusterNode:
    """
    Create a new ClusterNode.

    Each ClusterNode manages multiple :class:`~redis.asyncio.connection.Connection`
    objects for the (host, port).
    """
    __slots__ = ('_connections', '_free', 'connection_class', 'connection_kwargs', 'host', 'max_connections', 'name', 'port', 'response_callbacks', 'server_type')

    def __init__(self, host: str, port: Union[str, int], server_type: Optional[str]=None, *, max_connections: int=2 ** 31, connection_class: Type[Connection]=Connection, **connection_kwargs: Any) -> None:
        if host == 'localhost':
            host = socket.gethostbyname(host)
        connection_kwargs['host'] = host
        connection_kwargs['port'] = port
        self.host = host
        self.port = port
        self.name = get_node_name(host, port)
        self.server_type = server_type
        self.max_connections = max_connections
        self.connection_class = connection_class
        self.connection_kwargs = connection_kwargs
        self.response_callbacks = connection_kwargs.pop('response_callbacks', {})
        self._connections: List[Connection] = []
        self._free: Deque[Connection] = collections.deque(maxlen=self.max_connections)

    def __repr__(self) -> str:
        return f'[host={self.host}, port={self.port}, name={self.name}, server_type={self.server_type}]'

    def __eq__(self, obj: Any) -> bool:
        return isinstance(obj, ClusterNode) and obj.name == self.name
    _DEL_MESSAGE = 'Unclosed ClusterNode object'

    def __del__(self, _warn: Any=warnings.warn, _grl: Any=asyncio.get_running_loop) -> None:
        for connection in self._connections:
            if connection.is_connected:
                _warn(f'{self._DEL_MESSAGE} {self!r}', ResourceWarning, source=self)
                try:
                    context = {'client': self, 'message': self._DEL_MESSAGE}
                    _grl().call_exception_handler(context)
                except RuntimeError:
                    pass
                break

    async def disconnect(self) -> None:
        ret = await asyncio.gather(*(asyncio.create_task(connection.disconnect()) for connection in self._connections), return_exceptions=True)
        exc = next((res for res in ret if isinstance(res, Exception)), None)
        if exc:
            raise exc

    def acquire_connection(self) -> Connection:
        try:
            return self._free.popleft()
        except IndexError:
            if len(self._connections) < self.max_connections:
                connection = self.connection_class(**self.connection_kwargs)
                self._connections.append(connection)
                return connection
            raise MaxConnectionsError()

    async def parse_response(self, connection: Connection, command: str, **kwargs: Any) -> Any:
        try:
            if NEVER_DECODE in kwargs:
                response = await connection.read_response(disable_decoding=True)
                kwargs.pop(NEVER_DECODE)
            else:
                response = await connection.read_response()
        except ResponseError:
            if EMPTY_RESPONSE in kwargs:
                return kwargs[EMPTY_RESPONSE]
            raise
        if EMPTY_RESPONSE in kwargs:
            kwargs.pop(EMPTY_RESPONSE)
        if command in self.response_callbacks:
            return self.response_callbacks[command](response, **kwargs)
        return response

    async def execute_command(self, *args: Any, **kwargs: Any) -> Any:
        connection = self.acquire_connection()
        await connection.send_packed_command(connection.pack_command(*args), False)
        try:
            return await self.parse_response(connection, args[0], **kwargs)
        finally:
            self._free.append(connection)

    async def execute_pipeline(self, commands: List['PipelineCommand']) -> bool:
        connection = self.acquire_connection()
        await connection.send_packed_command(connection.pack_commands((cmd.args for cmd in commands)), False)
        ret = False
        for cmd in commands:
            try:
                cmd.result = await self.parse_response(connection, cmd.args[0], **cmd.kwargs)
            except Exception as e:
                cmd.result = e
                ret = True
        self._free.append(connection)
        return ret