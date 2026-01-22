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
class UnixDomainSocketConnection(AbstractConnection):
    """Manages UDS communication to and from a Redis server"""

    def __init__(self, *, path: str='', **kwargs):
        self.path = path
        super().__init__(**kwargs)

    def repr_pieces(self) -> Iterable[Tuple[str, Union[str, int]]]:
        pieces = [('path', self.path), ('db', self.db)]
        if self.client_name:
            pieces.append(('client_name', self.client_name))
        return pieces

    async def _connect(self):
        async with async_timeout(self.socket_connect_timeout):
            reader, writer = await asyncio.open_unix_connection(path=self.path)
        self._reader = reader
        self._writer = writer
        await self.on_connect()

    def _host_error(self) -> str:
        return self.path

    def _error_message(self, exception: BaseException) -> str:
        host_error = self._host_error()
        if len(exception.args) == 1:
            return f'Error connecting to unix socket: {host_error}. {exception.args[0]}.'
        else:
            return f'Error {exception.args[0]} connecting to unix socket: {host_error}. {exception.args[1]}.'