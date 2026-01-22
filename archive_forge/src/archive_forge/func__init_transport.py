from __future__ import annotations
import datetime
import enum
import logging
import typing
import warnings
from contextlib import asynccontextmanager, contextmanager
from types import TracebackType
from .__version__ import __version__
from ._auth import Auth, BasicAuth, FunctionAuth
from ._config import (
from ._decoders import SUPPORTED_DECODERS
from ._exceptions import (
from ._models import Cookies, Headers, Request, Response
from ._status_codes import codes
from ._transports.asgi import ASGITransport
from ._transports.base import AsyncBaseTransport, BaseTransport
from ._transports.default import AsyncHTTPTransport, HTTPTransport
from ._transports.wsgi import WSGITransport
from ._types import (
from ._urls import URL, QueryParams
from ._utils import (
def _init_transport(self, verify: VerifyTypes=True, cert: CertTypes | None=None, http1: bool=True, http2: bool=False, limits: Limits=DEFAULT_LIMITS, transport: AsyncBaseTransport | None=None, app: typing.Callable[..., typing.Any] | None=None, trust_env: bool=True) -> AsyncBaseTransport:
    if transport is not None:
        return transport
    if app is not None:
        return ASGITransport(app=app)
    return AsyncHTTPTransport(verify=verify, cert=cert, http1=http1, http2=http2, limits=limits, trust_env=trust_env)