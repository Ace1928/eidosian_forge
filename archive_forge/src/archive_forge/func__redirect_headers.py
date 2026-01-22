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
def _redirect_headers(self, request: Request, url: URL, method: str) -> Headers:
    """
        Return the headers that should be used for the redirect request.
        """
    headers = Headers(request.headers)
    if not same_origin(url, request.url):
        if not is_https_redirect(request.url, url):
            headers.pop('Authorization', None)
        headers['Host'] = url.netloc.decode('ascii')
    if method != request.method and method == 'GET':
        headers.pop('Content-Length', None)
        headers.pop('Transfer-Encoding', None)
    headers.pop('Cookie', None)
    return headers