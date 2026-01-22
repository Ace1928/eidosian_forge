from __future__ import annotations
import contextlib
import functools
import inspect
import re
import traceback
import types
import typing
import warnings
from contextlib import asynccontextmanager
from enum import Enum
from starlette._exception_handler import wrap_app_handling_exceptions
from starlette._utils import get_route_path, is_async_callable
from starlette.concurrency import run_in_threadpool
from starlette.convertors import CONVERTOR_TYPES, Convertor
from starlette.datastructures import URL, Headers, URLPath
from starlette.exceptions import HTTPException
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import PlainTextResponse, RedirectResponse, Response
from starlette.types import ASGIApp, Lifespan, Receive, Scope, Send
from starlette.websockets import WebSocket, WebSocketClose
def add_route(self, path: str, endpoint: typing.Callable[[Request], typing.Awaitable[Response] | Response], methods: list[str] | None=None, name: str | None=None, include_in_schema: bool=True) -> None:
    route = Route(path, endpoint=endpoint, methods=methods, name=name, include_in_schema=include_in_schema)
    self.routes.append(route)