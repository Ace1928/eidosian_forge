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
class _ProxyAdapter(httputil.HTTPMessageDelegate):

    def __init__(self, delegate: httputil.HTTPMessageDelegate, request_conn: httputil.HTTPConnection) -> None:
        self.connection = request_conn
        self.delegate = delegate

    def headers_received(self, start_line: Union[httputil.RequestStartLine, httputil.ResponseStartLine], headers: httputil.HTTPHeaders) -> Optional[Awaitable[None]]:
        self.connection.context._apply_xheaders(headers)
        return self.delegate.headers_received(start_line, headers)

    def data_received(self, chunk: bytes) -> Optional[Awaitable[None]]:
        return self.delegate.data_received(chunk)

    def finish(self) -> None:
        self.delegate.finish()
        self._cleanup()

    def on_connection_close(self) -> None:
        self.delegate.on_connection_close()
        self._cleanup()

    def _cleanup(self) -> None:
        self.connection.context._unapply_xheaders()