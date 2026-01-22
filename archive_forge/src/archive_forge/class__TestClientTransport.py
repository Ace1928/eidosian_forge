from __future__ import annotations
import contextlib
import inspect
import io
import json
import math
import queue
import sys
import typing
import warnings
from concurrent.futures import Future
from functools import cached_property
from types import GeneratorType
from urllib.parse import unquote, urljoin
import anyio
import anyio.abc
import anyio.from_thread
from anyio.abc import ObjectReceiveStream, ObjectSendStream
from anyio.streams.stapled import StapledObjectStream
from starlette._utils import is_async_callable
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from starlette.websockets import WebSocketDisconnect
class _TestClientTransport(httpx.BaseTransport):

    def __init__(self, app: ASGI3App, portal_factory: _PortalFactoryType, raise_server_exceptions: bool=True, root_path: str='', *, app_state: dict[str, typing.Any]) -> None:
        self.app = app
        self.raise_server_exceptions = raise_server_exceptions
        self.root_path = root_path
        self.portal_factory = portal_factory
        self.app_state = app_state

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        scheme = request.url.scheme
        netloc = request.url.netloc.decode(encoding='ascii')
        path = request.url.path
        raw_path = request.url.raw_path
        query = request.url.query.decode(encoding='ascii')
        default_port = {'http': 80, 'ws': 80, 'https': 443, 'wss': 443}[scheme]
        if ':' in netloc:
            host, port_string = netloc.split(':', 1)
            port = int(port_string)
        else:
            host = netloc
            port = default_port
        if 'host' in request.headers:
            headers: list[tuple[bytes, bytes]] = []
        elif port == default_port:
            headers = [(b'host', host.encode())]
        else:
            headers = [(b'host', f'{host}:{port}'.encode())]
        headers += [(key.lower().encode(), value.encode()) for key, value in request.headers.multi_items()]
        scope: dict[str, typing.Any]
        if scheme in {'ws', 'wss'}:
            subprotocol = request.headers.get('sec-websocket-protocol', None)
            if subprotocol is None:
                subprotocols: typing.Sequence[str] = []
            else:
                subprotocols = [value.strip() for value in subprotocol.split(',')]
            scope = {'type': 'websocket', 'path': unquote(path), 'raw_path': raw_path, 'root_path': self.root_path, 'scheme': scheme, 'query_string': query.encode(), 'headers': headers, 'client': None, 'server': [host, port], 'subprotocols': subprotocols, 'state': self.app_state.copy()}
            session = WebSocketTestSession(self.app, scope, self.portal_factory)
            raise _Upgrade(session)
        scope = {'type': 'http', 'http_version': '1.1', 'method': request.method, 'path': unquote(path), 'raw_path': raw_path, 'root_path': self.root_path, 'scheme': scheme, 'query_string': query.encode(), 'headers': headers, 'client': None, 'server': [host, port], 'extensions': {'http.response.debug': {}}, 'state': self.app_state.copy()}
        request_complete = False
        response_started = False
        response_complete: anyio.Event
        raw_kwargs: dict[str, typing.Any] = {'stream': io.BytesIO()}
        template = None
        context = None

        async def receive() -> Message:
            nonlocal request_complete
            if request_complete:
                if not response_complete.is_set():
                    await response_complete.wait()
                return {'type': 'http.disconnect'}
            body = request.read()
            if isinstance(body, str):
                body_bytes: bytes = body.encode('utf-8')
            elif body is None:
                body_bytes = b''
            elif isinstance(body, GeneratorType):
                try:
                    chunk = body.send(None)
                    if isinstance(chunk, str):
                        chunk = chunk.encode('utf-8')
                    return {'type': 'http.request', 'body': chunk, 'more_body': True}
                except StopIteration:
                    request_complete = True
                    return {'type': 'http.request', 'body': b''}
            else:
                body_bytes = body
            request_complete = True
            return {'type': 'http.request', 'body': body_bytes}

        async def send(message: Message) -> None:
            nonlocal raw_kwargs, response_started, template, context
            if message['type'] == 'http.response.start':
                assert not response_started, 'Received multiple "http.response.start" messages.'
                raw_kwargs['status_code'] = message['status']
                raw_kwargs['headers'] = [(key.decode(), value.decode()) for key, value in message.get('headers', [])]
                response_started = True
            elif message['type'] == 'http.response.body':
                assert response_started, 'Received "http.response.body" without "http.response.start".'
                assert not response_complete.is_set(), 'Received "http.response.body" after response completed.'
                body = message.get('body', b'')
                more_body = message.get('more_body', False)
                if request.method != 'HEAD':
                    raw_kwargs['stream'].write(body)
                if not more_body:
                    raw_kwargs['stream'].seek(0)
                    response_complete.set()
            elif message['type'] == 'http.response.debug':
                template = message['info']['template']
                context = message['info']['context']
        try:
            with self.portal_factory() as portal:
                response_complete = portal.call(anyio.Event)
                portal.call(self.app, scope, receive, send)
        except BaseException as exc:
            if self.raise_server_exceptions:
                raise exc
        if self.raise_server_exceptions:
            assert response_started, 'TestClient did not receive any response.'
        elif not response_started:
            raw_kwargs = {'status_code': 500, 'headers': [], 'stream': io.BytesIO()}
        raw_kwargs['stream'] = httpx.ByteStream(raw_kwargs['stream'].read())
        response = httpx.Response(**raw_kwargs, request=request)
        if template is not None:
            response.template = template
            response.context = context
        return response