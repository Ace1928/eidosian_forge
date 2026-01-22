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
class AbstractRoute(abc.ABC):

    def __init__(self, method: str, handler: Union[Handler, Type[AbstractView]], *, expect_handler: Optional[_ExpectHandler]=None, resource: Optional[AbstractResource]=None) -> None:
        if expect_handler is None:
            expect_handler = _default_expect_handler
        assert asyncio.iscoroutinefunction(expect_handler), f'Coroutine is expected, got {expect_handler!r}'
        method = method.upper()
        if not HTTP_METHOD_RE.match(method):
            raise ValueError(f'{method} is not allowed HTTP method')
        assert callable(handler), handler
        if asyncio.iscoroutinefunction(handler):
            pass
        elif inspect.isgeneratorfunction(handler):
            warnings.warn('Bare generators are deprecated, use @coroutine wrapper', DeprecationWarning)
        elif isinstance(handler, type) and issubclass(handler, AbstractView):
            pass
        else:
            warnings.warn('Bare functions are deprecated, use async ones', DeprecationWarning)

            @wraps(handler)
            async def handler_wrapper(request: Request) -> StreamResponse:
                result = old_handler(request)
                if asyncio.iscoroutine(result):
                    result = await result
                assert isinstance(result, StreamResponse)
                return result
            old_handler = handler
            handler = handler_wrapper
        self._method = method
        self._handler = handler
        self._expect_handler = expect_handler
        self._resource = resource

    @property
    def method(self) -> str:
        return self._method

    @property
    def handler(self) -> Handler:
        return self._handler

    @property
    @abc.abstractmethod
    def name(self) -> Optional[str]:
        """Optional route's name, always equals to resource's name."""

    @property
    def resource(self) -> Optional[AbstractResource]:
        return self._resource

    @abc.abstractmethod
    def get_info(self) -> _InfoDict:
        """Return a dict with additional info useful for introspection"""

    @abc.abstractmethod
    def url_for(self, *args: str, **kwargs: str) -> URL:
        """Construct url for route with additional params."""

    async def handle_expect_header(self, request: Request) -> Optional[StreamResponse]:
        return await self._expect_handler(request)