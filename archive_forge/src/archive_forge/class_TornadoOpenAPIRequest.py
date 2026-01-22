from __future__ import annotations
import json
import os
import sys
from http.cookies import SimpleCookie
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import tornado.httpclient
import tornado.web
from openapi_core import V30RequestValidator, V30ResponseValidator
from openapi_core.spec.paths import Spec
from openapi_core.validation.request.datatypes import RequestParameters
from tornado.httpclient import HTTPRequest, HTTPResponse
from werkzeug.datastructures import Headers, ImmutableMultiDict
from jupyterlab_server.spec import get_openapi_spec
class TornadoOpenAPIRequest:
    """
    Converts a torando request to an OpenAPI one
    """

    def __init__(self, request: HTTPRequest, spec: Spec):
        """Initialize the request."""
        self.request = request
        self.spec = spec
        if request.url is None:
            msg = 'Request URL is missing'
            raise RuntimeError(msg)
        self._url_parsed = urlparse(request.url)
        cookie: SimpleCookie = SimpleCookie()
        cookie.load(request.headers.get('Set-Cookie', ''))
        cookies = {}
        for key, morsel in cookie.items():
            cookies[key] = morsel.value
        o = urlparse(request.url)
        path: dict = {}
        self.parameters = RequestParameters(query=ImmutableMultiDict(parse_qs(o.query)), header=dict(request.headers), cookie=ImmutableMultiDict(cookies), path=path)

    @property
    def content_type(self) -> str:
        return 'application/json'

    @property
    def host_url(self) -> str:
        url = self.request.url
        return url[:url.index('/lab')]

    @property
    def path(self) -> str:
        url = None
        o = urlparse(self.request.url)
        for path_ in self.spec['paths']:
            if url:
                continue
            has_arg = '{' in path_
            path = path_[:path_.index('{')] if has_arg else path_
            if path in o.path:
                u = o.path[o.path.index(path):]
                if not has_arg and len(u) == len(path):
                    url = u
                if has_arg and (not u.endswith('/')):
                    url = u[:len(path)] + 'foo'
        if url is None:
            msg = f'Could not find matching pattern for {o.path}'
            raise ValueError(msg)
        return url

    @property
    def method(self) -> str:
        method = self.request.method
        return method and method.lower() or ''

    @property
    def body(self) -> bytes | None:
        if self.request.body is None:
            return None
        if not isinstance(self.request.body, bytes):
            msg = 'Request body is invalid'
            raise AssertionError(msg)
        return self.request.body

    @property
    def mimetype(self) -> str:
        request = self.request
        return request.headers.get('Content-Type') or request.headers.get('Accept') or 'application/json'