from __future__ import annotations
import json
import typing
from starlette import status
from starlette._utils import is_async_callable
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response
from starlette.types import Message, Receive, Scope, Send
from starlette.websockets import WebSocket
class HTTPEndpoint:

    def __init__(self, scope: Scope, receive: Receive, send: Send) -> None:
        assert scope['type'] == 'http'
        self.scope = scope
        self.receive = receive
        self.send = send
        self._allowed_methods = [method for method in ('GET', 'HEAD', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS') if getattr(self, method.lower(), None) is not None]

    def __await__(self) -> typing.Generator[typing.Any, None, None]:
        return self.dispatch().__await__()

    async def dispatch(self) -> None:
        request = Request(self.scope, receive=self.receive)
        handler_name = 'get' if request.method == 'HEAD' and (not hasattr(self, 'head')) else request.method.lower()
        handler: typing.Callable[[Request], typing.Any] = getattr(self, handler_name, self.method_not_allowed)
        is_async = is_async_callable(handler)
        if is_async:
            response = await handler(request)
        else:
            response = await run_in_threadpool(handler, request)
        await response(self.scope, self.receive, self.send)

    async def method_not_allowed(self, request: Request) -> Response:
        headers = {'Allow': ', '.join(self._allowed_methods)}
        if 'app' in self.scope:
            raise HTTPException(status_code=405, headers=headers)
        return PlainTextResponse('Method Not Allowed', status_code=405, headers=headers)