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
class Resource(AbstractResource):

    def __init__(self, *, name: Optional[str]=None) -> None:
        super().__init__(name=name)
        self._routes: List[ResourceRoute] = []

    def add_route(self, method: str, handler: Union[Type[AbstractView], Handler], *, expect_handler: Optional[_ExpectHandler]=None) -> 'ResourceRoute':
        for route_obj in self._routes:
            if route_obj.method == method or route_obj.method == hdrs.METH_ANY:
                raise RuntimeError('Added route will never be executed, method {route.method} is already registered'.format(route=route_obj))
        route_obj = ResourceRoute(method, handler, self, expect_handler=expect_handler)
        self.register_route(route_obj)
        return route_obj

    def register_route(self, route: 'ResourceRoute') -> None:
        assert isinstance(route, ResourceRoute), f'Instance of Route class is required, got {route!r}'
        self._routes.append(route)

    async def resolve(self, request: Request) -> _Resolve:
        allowed_methods: Set[str] = set()
        match_dict = self._match(request.rel_url.raw_path)
        if match_dict is None:
            return (None, allowed_methods)
        for route_obj in self._routes:
            route_method = route_obj.method
            allowed_methods.add(route_method)
            if route_method == request.method or route_method == hdrs.METH_ANY:
                return (UrlMappingMatchInfo(match_dict, route_obj), allowed_methods)
        else:
            return (None, allowed_methods)

    @abc.abstractmethod
    def _match(self, path: str) -> Optional[Dict[str, str]]:
        pass

    def __len__(self) -> int:
        return len(self._routes)

    def __iter__(self) -> Iterator['ResourceRoute']:
        return iter(self._routes)