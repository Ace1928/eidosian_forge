from __future__ import annotations
import asyncio
import email.utils
import functools
import http
import inspect
import logging
import socket
import warnings
from types import TracebackType
from typing import (
from ..datastructures import Headers, HeadersLike, MultipleValuesError
from ..exceptions import (
from ..extensions import Extension, ServerExtensionFactory
from ..extensions.permessage_deflate import enable_server_permessage_deflate
from ..headers import (
from ..http import USER_AGENT
from ..protocol import State
from ..typing import ExtensionHeader, LoggerLike, Origin, StatusLike, Subprotocol
from .compatibility import asyncio_timeout
from .handshake import build_response, check_request
from .http import read_request
from .protocol import WebSocketCommonProtocol
class Serve:
    """
    Start a WebSocket server listening on ``host`` and ``port``.

    Whenever a client connects, the server creates a
    :class:`WebSocketServerProtocol`, performs the opening handshake, and
    delegates to the connection handler, ``ws_handler``.

    The handler receives the :class:`WebSocketServerProtocol` and uses it to
    send and receive messages.

    Once the handler completes, either normally or with an exception, the
    server performs the closing handshake and closes the connection.

    Awaiting :func:`serve` yields a :class:`WebSocketServer`. This object
    provides a :meth:`~WebSocketServer.close` method to shut down the server::

        stop = asyncio.Future()  # set this future to exit the server

        server = await serve(...)
        await stop
        await server.close()

    :func:`serve` can be used as an asynchronous context manager. Then, the
    server is shut down automatically when exiting the context::

        stop = asyncio.Future()  # set this future to exit the server

        async with serve(...):
            await stop

    Args:
        ws_handler: Connection handler. It receives the WebSocket connection,
            which is a :class:`WebSocketServerProtocol`, in argument.
        host: Network interfaces the server binds to.
            See :meth:`~asyncio.loop.create_server` for details.
        port: TCP port the server listens on.
            See :meth:`~asyncio.loop.create_server` for details.
        create_protocol: Factory for the :class:`asyncio.Protocol` managing
            the connection. It defaults to :class:`WebSocketServerProtocol`.
            Set it to a wrapper or a subclass to customize connection handling.
        logger: Logger for this server.
            It defaults to ``logging.getLogger("websockets.server")``.
            See the :doc:`logging guide <../../topics/logging>` for details.
        compression: The "permessage-deflate" extension is enabled by default.
            Set ``compression`` to :obj:`None` to disable it. See the
            :doc:`compression guide <../../topics/compression>` for details.
        origins: Acceptable values of the ``Origin`` header, for defending
            against Cross-Site WebSocket Hijacking attacks. Include :obj:`None`
            in the list if the lack of an origin is acceptable.
        extensions: List of supported extensions, in order in which they
            should be negotiated and run.
        subprotocols: List of supported subprotocols, in order of decreasing
            preference.
        extra_headers (Union[HeadersLike, Callable[[str, Headers], HeadersLike]]):
            Arbitrary HTTP headers to add to the response. This can be
            a :data:`~websockets.datastructures.HeadersLike` or a callable
            taking the request path and headers in arguments and returning
            a :data:`~websockets.datastructures.HeadersLike`.
        server_header: Value of  the ``Server`` response header.
            It defaults to ``"Python/x.y.z websockets/X.Y"``.
            Setting it to :obj:`None` removes the header.
        process_request (Optional[Callable[[str, Headers],             Awaitable[Optional[Tuple[StatusLike, HeadersLike, bytes]]]]]):
            Intercept HTTP request before the opening handshake.
            See :meth:`~WebSocketServerProtocol.process_request` for details.
        select_subprotocol: Select a subprotocol supported by the client.
            See :meth:`~WebSocketServerProtocol.select_subprotocol` for details.
        open_timeout: Timeout for opening connections in seconds.
            :obj:`None` disables the timeout.

    See :class:`~websockets.legacy.protocol.WebSocketCommonProtocol` for the
    documentation of ``ping_interval``, ``ping_timeout``, ``close_timeout``,
    ``max_size``, ``max_queue``, ``read_limit``, and ``write_limit``.

    Any other keyword arguments are passed the event loop's
    :meth:`~asyncio.loop.create_server` method.

    For example:

    * You can set ``ssl`` to a :class:`~ssl.SSLContext` to enable TLS.

    * You can set ``sock`` to a :obj:`~socket.socket` that you created
      outside of websockets.

    Returns:
        WebSocketServer: WebSocket server.

    """

    def __init__(self, ws_handler: Union[Callable[[WebSocketServerProtocol], Awaitable[Any]], Callable[[WebSocketServerProtocol, str], Awaitable[Any]]], host: Optional[Union[str, Sequence[str]]]=None, port: Optional[int]=None, *, create_protocol: Optional[Callable[..., WebSocketServerProtocol]]=None, logger: Optional[LoggerLike]=None, compression: Optional[str]='deflate', origins: Optional[Sequence[Optional[Origin]]]=None, extensions: Optional[Sequence[ServerExtensionFactory]]=None, subprotocols: Optional[Sequence[Subprotocol]]=None, extra_headers: Optional[HeadersLikeOrCallable]=None, server_header: Optional[str]=USER_AGENT, process_request: Optional[Callable[[str, Headers], Awaitable[Optional[HTTPResponse]]]]=None, select_subprotocol: Optional[Callable[[Sequence[Subprotocol], Sequence[Subprotocol]], Subprotocol]]=None, open_timeout: Optional[float]=10, ping_interval: Optional[float]=20, ping_timeout: Optional[float]=20, close_timeout: Optional[float]=None, max_size: Optional[int]=2 ** 20, max_queue: Optional[int]=2 ** 5, read_limit: int=2 ** 16, write_limit: int=2 ** 16, **kwargs: Any) -> None:
        timeout: Optional[float] = kwargs.pop('timeout', None)
        if timeout is None:
            timeout = 10
        else:
            warnings.warn('rename timeout to close_timeout', DeprecationWarning)
        if close_timeout is None:
            close_timeout = timeout
        klass: Optional[Type[WebSocketServerProtocol]] = kwargs.pop('klass', None)
        if klass is None:
            klass = WebSocketServerProtocol
        else:
            warnings.warn('rename klass to create_protocol', DeprecationWarning)
        if create_protocol is None:
            create_protocol = klass
        legacy_recv: bool = kwargs.pop('legacy_recv', False)
        _loop: Optional[asyncio.AbstractEventLoop] = kwargs.pop('loop', None)
        if _loop is None:
            loop = asyncio.get_event_loop()
        else:
            loop = _loop
            warnings.warn('remove loop argument', DeprecationWarning)
        ws_server = WebSocketServer(logger=logger)
        secure = kwargs.get('ssl') is not None
        if compression == 'deflate':
            extensions = enable_server_permessage_deflate(extensions)
        elif compression is not None:
            raise ValueError(f'unsupported compression: {compression}')
        if subprotocols is not None:
            validate_subprotocols(subprotocols)
        factory = functools.partial(create_protocol, remove_path_argument(ws_handler), ws_server, host=host, port=port, secure=secure, open_timeout=open_timeout, ping_interval=ping_interval, ping_timeout=ping_timeout, close_timeout=close_timeout, max_size=max_size, max_queue=max_queue, read_limit=read_limit, write_limit=write_limit, loop=_loop, legacy_recv=legacy_recv, origins=origins, extensions=extensions, subprotocols=subprotocols, extra_headers=extra_headers, server_header=server_header, process_request=process_request, select_subprotocol=select_subprotocol, logger=logger)
        if kwargs.pop('unix', False):
            path: Optional[str] = kwargs.pop('path', None)
            assert host is None and port is None
            create_server = functools.partial(loop.create_unix_server, factory, path, **kwargs)
        else:
            create_server = functools.partial(loop.create_server, factory, host, port, **kwargs)
        self._create_server = create_server
        self.ws_server = ws_server

    async def __aenter__(self) -> WebSocketServer:
        return await self

    async def __aexit__(self, exc_type: Optional[Type[BaseException]], exc_value: Optional[BaseException], traceback: Optional[TracebackType]) -> None:
        self.ws_server.close()
        await self.ws_server.wait_closed()

    def __await__(self) -> Generator[Any, None, WebSocketServer]:
        return self.__await_impl__().__await__()

    async def __await_impl__(self) -> WebSocketServer:
        server = await self._create_server()
        self.ws_server.wrap(server)
        return self.ws_server
    __iter__ = __await__