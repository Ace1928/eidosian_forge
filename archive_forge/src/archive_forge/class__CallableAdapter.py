import socket
import ssl
from tornado.escape import native_str
from tornado.http1connection import HTTP1ServerConnection, HTTP1ConnectionParameters
from tornado import httputil
from tornado import iostream
from tornado import netutil
from tornado.tcpserver import TCPServer
from tornado.util import Configurable
import typing
from typing import Union, Any, Dict, Callable, List, Type, Tuple, Optional, Awaitable
class _CallableAdapter(httputil.HTTPMessageDelegate):

    def __init__(self, request_callback: Callable[[httputil.HTTPServerRequest], None], request_conn: httputil.HTTPConnection) -> None:
        self.connection = request_conn
        self.request_callback = request_callback
        self.request = None
        self.delegate = None
        self._chunks = []

    def headers_received(self, start_line: Union[httputil.RequestStartLine, httputil.ResponseStartLine], headers: httputil.HTTPHeaders) -> Optional[Awaitable[None]]:
        self.request = httputil.HTTPServerRequest(connection=self.connection, start_line=typing.cast(httputil.RequestStartLine, start_line), headers=headers)
        return None

    def data_received(self, chunk: bytes) -> Optional[Awaitable[None]]:
        self._chunks.append(chunk)
        return None

    def finish(self) -> None:
        assert self.request is not None
        self.request.body = b''.join(self._chunks)
        self.request._parse_body()
        self.request_callback(self.request)

    def on_connection_close(self) -> None:
        del self._chunks