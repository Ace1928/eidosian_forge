import logging
import ssl
from base64 import b64encode
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple, Union
from .._backends.base import SOCKET_OPTION, AsyncNetworkBackend
from .._exceptions import ProxyError
from .._models import (
from .._ssl import default_ssl_context
from .._synchronization import AsyncLock
from .._trace import Trace
from .connection import AsyncHTTPConnection
from .connection_pool import AsyncConnectionPool
from .http11 import AsyncHTTP11Connection
from .interfaces import AsyncConnectionInterface
class AsyncTunnelHTTPConnection(AsyncConnectionInterface):

    def __init__(self, proxy_origin: Origin, remote_origin: Origin, ssl_context: Optional[ssl.SSLContext]=None, proxy_ssl_context: Optional[ssl.SSLContext]=None, proxy_headers: Optional[Sequence[Tuple[bytes, bytes]]]=None, keepalive_expiry: Optional[float]=None, http1: bool=True, http2: bool=False, network_backend: Optional[AsyncNetworkBackend]=None, socket_options: Optional[Iterable[SOCKET_OPTION]]=None) -> None:
        self._connection: AsyncConnectionInterface = AsyncHTTPConnection(origin=proxy_origin, keepalive_expiry=keepalive_expiry, network_backend=network_backend, socket_options=socket_options, ssl_context=proxy_ssl_context)
        self._proxy_origin = proxy_origin
        self._remote_origin = remote_origin
        self._ssl_context = ssl_context
        self._proxy_ssl_context = proxy_ssl_context
        self._proxy_headers = enforce_headers(proxy_headers, name='proxy_headers')
        self._keepalive_expiry = keepalive_expiry
        self._http1 = http1
        self._http2 = http2
        self._connect_lock = AsyncLock()
        self._connected = False

    async def handle_async_request(self, request: Request) -> Response:
        timeouts = request.extensions.get('timeout', {})
        timeout = timeouts.get('connect', None)
        async with self._connect_lock:
            if not self._connected:
                target = b'%b:%d' % (self._remote_origin.host, self._remote_origin.port)
                connect_url = URL(scheme=self._proxy_origin.scheme, host=self._proxy_origin.host, port=self._proxy_origin.port, target=target)
                connect_headers = merge_headers([(b'Host', target), (b'Accept', b'*/*')], self._proxy_headers)
                connect_request = Request(method=b'CONNECT', url=connect_url, headers=connect_headers, extensions=request.extensions)
                connect_response = await self._connection.handle_async_request(connect_request)
                if connect_response.status < 200 or connect_response.status > 299:
                    reason_bytes = connect_response.extensions.get('reason_phrase', b'')
                    reason_str = reason_bytes.decode('ascii', errors='ignore')
                    msg = '%d %s' % (connect_response.status, reason_str)
                    await self._connection.aclose()
                    raise ProxyError(msg)
                stream = connect_response.extensions['network_stream']
                ssl_context = default_ssl_context() if self._ssl_context is None else self._ssl_context
                alpn_protocols = ['http/1.1', 'h2'] if self._http2 else ['http/1.1']
                ssl_context.set_alpn_protocols(alpn_protocols)
                kwargs = {'ssl_context': ssl_context, 'server_hostname': self._remote_origin.host.decode('ascii'), 'timeout': timeout}
                async with Trace('start_tls', logger, request, kwargs) as trace:
                    stream = await stream.start_tls(**kwargs)
                    trace.return_value = stream
                ssl_object = stream.get_extra_info('ssl_object')
                http2_negotiated = ssl_object is not None and ssl_object.selected_alpn_protocol() == 'h2'
                if http2_negotiated or (self._http2 and (not self._http1)):
                    from .http2 import AsyncHTTP2Connection
                    self._connection = AsyncHTTP2Connection(origin=self._remote_origin, stream=stream, keepalive_expiry=self._keepalive_expiry)
                else:
                    self._connection = AsyncHTTP11Connection(origin=self._remote_origin, stream=stream, keepalive_expiry=self._keepalive_expiry)
                self._connected = True
        return await self._connection.handle_async_request(request)

    def can_handle_request(self, origin: Origin) -> bool:
        return origin == self._remote_origin

    async def aclose(self) -> None:
        await self._connection.aclose()

    def info(self) -> str:
        return self._connection.info()

    def is_available(self) -> bool:
        return self._connection.is_available()

    def has_expired(self) -> bool:
        return self._connection.has_expired()

    def is_idle(self) -> bool:
        return self._connection.is_idle()

    def is_closed(self) -> bool:
        return self._connection.is_closed()

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} [{self.info()}]>'