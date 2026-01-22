from __future__ import annotations
import asyncio
import functools
import logging
import random
import urllib.parse
import warnings
from types import TracebackType
from typing import (
from ..datastructures import Headers, HeadersLike
from ..exceptions import (
from ..extensions import ClientExtensionFactory, Extension
from ..extensions.permessage_deflate import enable_client_permessage_deflate
from ..headers import (
from ..http import USER_AGENT
from ..typing import ExtensionHeader, LoggerLike, Origin, Subprotocol
from ..uri import WebSocketURI, parse_uri
from .compatibility import asyncio_timeout
from .handshake import build_request, check_response
from .http import read_response
from .protocol import WebSocketCommonProtocol
class WebSocketClientProtocol(WebSocketCommonProtocol):
    """
    WebSocket client connection.

    :class:`WebSocketClientProtocol` provides :meth:`recv` and :meth:`send`
    coroutines for receiving and sending messages.

    It supports asynchronous iteration to receive incoming messages::

        async for message in websocket:
            await process(message)

    The iterator exits normally when the connection is closed with close code
    1000 (OK) or 1001 (going away) or without a close code. It raises
    a :exc:`~websockets.exceptions.ConnectionClosedError` when the connection
    is closed with any other code.

    See :func:`connect` for the documentation of ``logger``, ``origin``,
    ``extensions``, ``subprotocols``, ``extra_headers``, and
    ``user_agent_header``.

    See :class:`~websockets.legacy.protocol.WebSocketCommonProtocol` for the
    documentation of ``ping_interval``, ``ping_timeout``, ``close_timeout``,
    ``max_size``, ``max_queue``, ``read_limit``, and ``write_limit``.

    """
    is_client = True
    side = 'client'

    def __init__(self, *, logger: Optional[LoggerLike]=None, origin: Optional[Origin]=None, extensions: Optional[Sequence[ClientExtensionFactory]]=None, subprotocols: Optional[Sequence[Subprotocol]]=None, extra_headers: Optional[HeadersLike]=None, user_agent_header: Optional[str]=USER_AGENT, **kwargs: Any) -> None:
        if logger is None:
            logger = logging.getLogger('websockets.client')
        super().__init__(logger=logger, **kwargs)
        self.origin = origin
        self.available_extensions = extensions
        self.available_subprotocols = subprotocols
        self.extra_headers = extra_headers
        self.user_agent_header = user_agent_header

    def write_http_request(self, path: str, headers: Headers) -> None:
        """
        Write request line and headers to the HTTP request.

        """
        self.path = path
        self.request_headers = headers
        if self.debug:
            self.logger.debug('> GET %s HTTP/1.1', path)
            for key, value in headers.raw_items():
                self.logger.debug('> %s: %s', key, value)
        request = f'GET {path} HTTP/1.1\r\n'
        request += str(headers)
        self.transport.write(request.encode())

    async def read_http_response(self) -> Tuple[int, Headers]:
        """
        Read status line and headers from the HTTP response.

        If the response contains a body, it may be read from ``self.reader``
        after this coroutine returns.

        Raises:
            InvalidMessage: If the HTTP message is malformed or isn't an
                HTTP/1.1 GET response.

        """
        try:
            status_code, reason, headers = await read_response(self.reader)
        except Exception as exc:
            raise InvalidMessage('did not receive a valid HTTP response') from exc
        if self.debug:
            self.logger.debug('< HTTP/1.1 %d %s', status_code, reason)
            for key, value in headers.raw_items():
                self.logger.debug('< %s: %s', key, value)
        self.response_headers = headers
        return (status_code, self.response_headers)

    @staticmethod
    def process_extensions(headers: Headers, available_extensions: Optional[Sequence[ClientExtensionFactory]]) -> List[Extension]:
        """
        Handle the Sec-WebSocket-Extensions HTTP response header.

        Check that each extension is supported, as well as its parameters.

        Return the list of accepted extensions.

        Raise :exc:`~websockets.exceptions.InvalidHandshake` to abort the
        connection.

        :rfc:`6455` leaves the rules up to the specification of each
        :extension.

        To provide this level of flexibility, for each extension accepted by
        the server, we check for a match with each extension available in the
        client configuration. If no match is found, an exception is raised.

        If several variants of the same extension are accepted by the server,
        it may be configured several times, which won't make sense in general.
        Extensions must implement their own requirements. For this purpose,
        the list of previously accepted extensions is provided.

        Other requirements, for example related to mandatory extensions or the
        order of extensions, may be implemented by overriding this method.

        """
        accepted_extensions: List[Extension] = []
        header_values = headers.get_all('Sec-WebSocket-Extensions')
        if header_values:
            if available_extensions is None:
                raise InvalidHandshake('no extensions supported')
            parsed_header_values: List[ExtensionHeader] = sum([parse_extension(header_value) for header_value in header_values], [])
            for name, response_params in parsed_header_values:
                for extension_factory in available_extensions:
                    if extension_factory.name != name:
                        continue
                    try:
                        extension = extension_factory.process_response_params(response_params, accepted_extensions)
                    except NegotiationError:
                        continue
                    accepted_extensions.append(extension)
                    break
                else:
                    raise NegotiationError(f'Unsupported extension: name = {name}, params = {response_params}')
        return accepted_extensions

    @staticmethod
    def process_subprotocol(headers: Headers, available_subprotocols: Optional[Sequence[Subprotocol]]) -> Optional[Subprotocol]:
        """
        Handle the Sec-WebSocket-Protocol HTTP response header.

        Check that it contains exactly one supported subprotocol.

        Return the selected subprotocol.

        """
        subprotocol: Optional[Subprotocol] = None
        header_values = headers.get_all('Sec-WebSocket-Protocol')
        if header_values:
            if available_subprotocols is None:
                raise InvalidHandshake('no subprotocols supported')
            parsed_header_values: Sequence[Subprotocol] = sum([parse_subprotocol(header_value) for header_value in header_values], [])
            if len(parsed_header_values) > 1:
                subprotocols = ', '.join(parsed_header_values)
                raise InvalidHandshake(f'multiple subprotocols: {subprotocols}')
            subprotocol = parsed_header_values[0]
            if subprotocol not in available_subprotocols:
                raise NegotiationError(f'unsupported subprotocol: {subprotocol}')
        return subprotocol

    async def handshake(self, wsuri: WebSocketURI, origin: Optional[Origin]=None, available_extensions: Optional[Sequence[ClientExtensionFactory]]=None, available_subprotocols: Optional[Sequence[Subprotocol]]=None, extra_headers: Optional[HeadersLike]=None) -> None:
        """
        Perform the client side of the opening handshake.

        Args:
            wsuri: URI of the WebSocket server.
            origin: Value of the ``Origin`` header.
            extensions: List of supported extensions, in order in which they
                should be negotiated and run.
            subprotocols: List of supported subprotocols, in order of decreasing
                preference.
            extra_headers: Arbitrary HTTP headers to add to the handshake request.

        Raises:
            InvalidHandshake: If the handshake fails.

        """
        request_headers = Headers()
        request_headers['Host'] = build_host(wsuri.host, wsuri.port, wsuri.secure)
        if wsuri.user_info:
            request_headers['Authorization'] = build_authorization_basic(*wsuri.user_info)
        if origin is not None:
            request_headers['Origin'] = origin
        key = build_request(request_headers)
        if available_extensions is not None:
            extensions_header = build_extension([(extension_factory.name, extension_factory.get_request_params()) for extension_factory in available_extensions])
            request_headers['Sec-WebSocket-Extensions'] = extensions_header
        if available_subprotocols is not None:
            protocol_header = build_subprotocol(available_subprotocols)
            request_headers['Sec-WebSocket-Protocol'] = protocol_header
        if self.extra_headers is not None:
            request_headers.update(self.extra_headers)
        if self.user_agent_header is not None:
            request_headers.setdefault('User-Agent', self.user_agent_header)
        self.write_http_request(wsuri.resource_name, request_headers)
        status_code, response_headers = await self.read_http_response()
        if status_code in (301, 302, 303, 307, 308):
            if 'Location' not in response_headers:
                raise InvalidHeader('Location')
            raise RedirectHandshake(response_headers['Location'])
        elif status_code != 101:
            raise InvalidStatusCode(status_code, response_headers)
        check_response(response_headers, key)
        self.extensions = self.process_extensions(response_headers, available_extensions)
        self.subprotocol = self.process_subprotocol(response_headers, available_subprotocols)
        self.connection_open()