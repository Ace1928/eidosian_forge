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
def _get_proxy_map(self, proxies: ProxiesTypes | None, allow_env_proxies: bool) -> dict[str, Proxy | None]:
    if proxies is None:
        if allow_env_proxies:
            return {key: None if url is None else Proxy(url=url) for key, url in get_environment_proxies().items()}
        return {}
    if isinstance(proxies, dict):
        new_proxies = {}
        for key, value in proxies.items():
            proxy = Proxy(url=value) if isinstance(value, (str, URL)) else value
            new_proxies[str(key)] = proxy
        return new_proxies
    else:
        proxy = Proxy(url=proxies) if isinstance(proxies, (str, URL)) else proxies
        return {'all://': proxy}