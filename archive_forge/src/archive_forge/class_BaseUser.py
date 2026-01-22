from __future__ import annotations
import functools
import inspect
import sys
import typing
from urllib.parse import urlencode
from starlette._utils import is_async_callable
from starlette.exceptions import HTTPException
from starlette.requests import HTTPConnection, Request
from starlette.responses import RedirectResponse
from starlette.websockets import WebSocket
class BaseUser:

    @property
    def is_authenticated(self) -> bool:
        raise NotImplementedError()

    @property
    def display_name(self) -> str:
        raise NotImplementedError()

    @property
    def identity(self) -> str:
        raise NotImplementedError()