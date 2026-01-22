import asyncio
import socket
import sys
from typing import Callable, List, Optional, Union
from redis.compat import TypedDict
from ..exceptions import ConnectionError, InvalidResponse, RedisError
from ..typing import EncodableT
from ..utils import HIREDIS_AVAILABLE
from .base import AsyncBaseParser, BaseParser
from .socket import (
class _AsyncHiredisParser(AsyncBaseParser):
    """Async implementation of parser class for connections using Hiredis"""
    __slots__ = ('_reader',)

    def __init__(self, socket_read_size: int):
        if not HIREDIS_AVAILABLE:
            raise RedisError('Hiredis is not available.')
        super().__init__(socket_read_size=socket_read_size)
        self._reader = None

    def on_connect(self, connection):
        import hiredis
        self._stream = connection._reader
        kwargs: _HiredisReaderArgs = {'protocolError': InvalidResponse, 'replyError': self.parse_error}
        if connection.encoder.decode_responses:
            kwargs['encoding'] = connection.encoder.encoding
            kwargs['errors'] = connection.encoder.encoding_errors
        self._reader = hiredis.Reader(**kwargs)
        self._connected = True

    def on_disconnect(self):
        self._connected = False

    async def can_read_destructive(self):
        if not self._connected:
            raise ConnectionError(SERVER_CLOSED_CONNECTION_ERROR)
        if self._reader.gets():
            return True
        try:
            async with async_timeout(0):
                return await self.read_from_socket()
        except asyncio.TimeoutError:
            return False

    async def read_from_socket(self):
        buffer = await self._stream.read(self._read_size)
        if not buffer or not isinstance(buffer, bytes):
            raise ConnectionError(SERVER_CLOSED_CONNECTION_ERROR) from None
        self._reader.feed(buffer)
        return True

    async def read_response(self, disable_decoding: bool=False) -> Union[EncodableT, List[EncodableT]]:
        if not self._connected:
            raise ConnectionError(SERVER_CLOSED_CONNECTION_ERROR) from None
        if disable_decoding:
            response = self._reader.gets(False)
        else:
            response = self._reader.gets()
        while response is False:
            await self.read_from_socket()
            if disable_decoding:
                response = self._reader.gets(False)
            else:
                response = self._reader.gets()
        if isinstance(response, ConnectionError):
            raise response
        elif isinstance(response, list) and response and isinstance(response[0], ConnectionError):
            raise response[0]
        return response