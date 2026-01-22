from __future__ import annotations
import sys
import typing
import warnings
from starlette.datastructures import State, URLPath
from starlette.middleware import Middleware, _MiddlewareClass
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.errors import ServerErrorMiddleware
from starlette.middleware.exceptions import ExceptionMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import BaseRoute, Router
from starlette.types import ASGIApp, ExceptionHandler, Lifespan, Receive, Scope, Send
from starlette.websockets import WebSocket
def add_middleware(self, middleware_class: typing.Type[_MiddlewareClass[P]], *args: P.args, **kwargs: P.kwargs) -> None:
    if self.middleware_stack is not None:
        raise RuntimeError('Cannot add middleware after an application has started')
    self.user_middleware.insert(0, Middleware(middleware_class, *args, **kwargs))