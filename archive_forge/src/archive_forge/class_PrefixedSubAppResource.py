import abc
import asyncio
import base64
import hashlib
import inspect
import keyword
import os
import re
import warnings
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from types import MappingProxyType
from typing import (
from yarl import URL, __version__ as yarl_version  # type: ignore[attr-defined]
from . import hdrs
from .abc import AbstractMatchInfo, AbstractRouter, AbstractView
from .helpers import DEBUG
from .http import HttpVersion11
from .typedefs import Handler, PathLike
from .web_exceptions import (
from .web_fileresponse import FileResponse
from .web_request import Request
from .web_response import Response, StreamResponse
from .web_routedef import AbstractRouteDef
class PrefixedSubAppResource(PrefixResource):

    def __init__(self, prefix: str, app: 'Application') -> None:
        super().__init__(prefix)
        self._app = app
        for resource in app.router.resources():
            resource.add_prefix(prefix)

    def add_prefix(self, prefix: str) -> None:
        super().add_prefix(prefix)
        for resource in self._app.router.resources():
            resource.add_prefix(prefix)

    def url_for(self, *args: str, **kwargs: str) -> URL:
        raise RuntimeError('.url_for() is not supported by sub-application root')

    def get_info(self) -> _InfoDict:
        return {'app': self._app, 'prefix': self._prefix}

    async def resolve(self, request: Request) -> _Resolve:
        if not request.url.raw_path.startswith(self._prefix2) and request.url.raw_path != self._prefix:
            return (None, set())
        match_info = await self._app.router.resolve(request)
        match_info.add_app(self._app)
        if isinstance(match_info.http_exception, HTTPMethodNotAllowed):
            methods = match_info.http_exception.allowed_methods
        else:
            methods = set()
        return (match_info, methods)

    def __len__(self) -> int:
        return len(self._app.router.routes())

    def __iter__(self) -> Iterator[AbstractRoute]:
        return iter(self._app.router.routes())

    def __repr__(self) -> str:
        return '<PrefixedSubAppResource {prefix} -> {app!r}>'.format(prefix=self._prefix, app=self._app)