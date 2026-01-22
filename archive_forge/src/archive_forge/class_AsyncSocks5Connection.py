import logging
import ssl
import typing
from socksio import socks5
from .._backends.auto import AutoBackend
from .._backends.base import AsyncNetworkBackend, AsyncNetworkStream
from .._exceptions import ConnectionNotAvailable, ProxyError
from .._models import URL, Origin, Request, Response, enforce_bytes, enforce_url
from .._ssl import default_ssl_context
from .._synchronization import AsyncLock
from .._trace import Trace
from .connection_pool import AsyncConnectionPool
from .http11 import AsyncHTTP11Connection
from .interfaces import AsyncConnectionInterface
class AsyncSocks5Connection(AsyncConnectionInterface):

    def __init__(self, proxy_origin: Origin, remote_origin: Origin, proxy_auth: typing.Optional[typing.Tuple[bytes, bytes]]=None, ssl_context: typing.Optional[ssl.SSLContext]=None, keepalive_expiry: typing.Optional[float]=None, http1: bool=True, http2: bool=False, network_backend: typing.Optional[AsyncNetworkBackend]=None) -> None:
        self._proxy_origin = proxy_origin
        self._remote_origin = remote_origin
        self._proxy_auth = proxy_auth
        self._ssl_context = ssl_context
        self._keepalive_expiry = keepalive_expiry
        self._http1 = http1
        self._http2 = http2
        self._network_backend: AsyncNetworkBackend = AutoBackend() if network_backend is None else network_backend
        self._connect_lock = AsyncLock()
        self._connection: typing.Optional[AsyncConnectionInterface] = None
        self._connect_failed = False

    async def handle_async_request(self, request: Request) -> Response:
        timeouts = request.extensions.get('timeout', {})
        sni_hostname = request.extensions.get('sni_hostname', None)
        timeout = timeouts.get('connect', None)
        async with self._connect_lock:
            if self._connection is None:
                try:
                    kwargs = {'host': self._proxy_origin.host.decode('ascii'), 'port': self._proxy_origin.port, 'timeout': timeout}
                    async with Trace('connect_tcp', logger, request, kwargs) as trace:
                        stream = await self._network_backend.connect_tcp(**kwargs)
                        trace.return_value = stream
                    kwargs = {'stream': stream, 'host': self._remote_origin.host.decode('ascii'), 'port': self._remote_origin.port, 'auth': self._proxy_auth}
                    async with Trace('setup_socks5_connection', logger, request, kwargs) as trace:
                        await _init_socks5_connection(**kwargs)
                        trace.return_value = stream
                    if self._remote_origin.scheme == b'https':
                        ssl_context = default_ssl_context() if self._ssl_context is None else self._ssl_context
                        alpn_protocols = ['http/1.1', 'h2'] if self._http2 else ['http/1.1']
                        ssl_context.set_alpn_protocols(alpn_protocols)
                        kwargs = {'ssl_context': ssl_context, 'server_hostname': sni_hostname or self._remote_origin.host.decode('ascii'), 'timeout': timeout}
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
                except Exception as exc:
                    self._connect_failed = True
                    raise exc
            elif not self._connection.is_available():
                raise ConnectionNotAvailable()
        return await self._connection.handle_async_request(request)

    def can_handle_request(self, origin: Origin) -> bool:
        return origin == self._remote_origin

    async def aclose(self) -> None:
        if self._connection is not None:
            await self._connection.aclose()

    def is_available(self) -> bool:
        if self._connection is None:
            return self._http2 and (self._remote_origin.scheme == b'https' or not self._http1) and (not self._connect_failed)
        return self._connection.is_available()

    def has_expired(self) -> bool:
        if self._connection is None:
            return self._connect_failed
        return self._connection.has_expired()

    def is_idle(self) -> bool:
        if self._connection is None:
            return self._connect_failed
        return self._connection.is_idle()

    def is_closed(self) -> bool:
        if self._connection is None:
            return self._connect_failed
        return self._connection.is_closed()

    def info(self) -> str:
        if self._connection is None:
            return 'CONNECTION FAILED' if self._connect_failed else 'CONNECTING'
        return self._connection.info()

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} [{self.info()}]>'