import ssl
import typing
from typing import Optional
from .._exceptions import ReadError
from .base import (
class MockStream(NetworkStream):

    def __init__(self, buffer: typing.List[bytes], http2: bool=False) -> None:
        self._buffer = buffer
        self._http2 = http2
        self._closed = False

    def read(self, max_bytes: int, timeout: Optional[float]=None) -> bytes:
        if self._closed:
            raise ReadError('Connection closed')
        if not self._buffer:
            return b''
        return self._buffer.pop(0)

    def write(self, buffer: bytes, timeout: Optional[float]=None) -> None:
        pass

    def close(self) -> None:
        self._closed = True

    def start_tls(self, ssl_context: ssl.SSLContext, server_hostname: Optional[str]=None, timeout: Optional[float]=None) -> NetworkStream:
        return self

    def get_extra_info(self, info: str) -> typing.Any:
        return MockSSLObject(http2=self._http2) if info == 'ssl_object' else None

    def __repr__(self) -> str:
        return '<httpcore.MockStream>'