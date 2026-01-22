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
class AuthCredentials:

    def __init__(self, scopes: typing.Sequence[str] | None=None):
        self.scopes = [] if scopes is None else list(scopes)