from __future__ import annotations
import collections
import os
import warnings
import zlib
from dataclasses import dataclass
from functools import wraps
from http.cookiejar import CookieJar
from typing import TYPE_CHECKING, Iterable, Optional
from urllib.parse import urldefrag, urljoin, urlunparse as _urlunparse
from zope.interface import implementer
from incremental import Version
from twisted.internet import defer, protocol, task
from twisted.internet.abstract import isIPv6Address
from twisted.internet.defer import Deferred
from twisted.internet.endpoints import HostnameEndpoint, wrapClientTLS
from twisted.internet.interfaces import IOpenSSLContextFactory, IProtocol
from twisted.logger import Logger
from twisted.python.compat import nativeString, networkString
from twisted.python.components import proxyForInterface
from twisted.python.deprecate import (
from twisted.python.failure import Failure
from twisted.web import error, http
from twisted.web._newclient import _ensureValidMethod, _ensureValidURI
from twisted.web.http_headers import Headers
from twisted.web.iweb import (
from twisted.web._newclient import (
from twisted.web.error import SchemeNotSupported
@implementer(IAgent)
class CookieAgent:
    """
    L{CookieAgent} extends the basic L{Agent} to add RFC-compliant handling of
    HTTP cookies.  Cookies are written to and extracted from a L{CookieJar}
    instance.

    The same cookie jar instance will be used for any requests through this
    agent, mutating it whenever a I{Set-Cookie} header appears in a response.

    @ivar _agent: Underlying Twisted Web agent to issue requests through.

    @ivar cookieJar: Initialized cookie jar to read cookies from and store
        cookies to.

    @since: 11.1
    """
    _agent: IAgent
    cookieJar: CookieJar

    def __init__(self, agent: IAgent, cookieJar: CookieJar) -> None:
        self._agent = agent
        self.cookieJar = cookieJar

    def request(self, method: bytes, uri: bytes, headers: Optional[Headers]=None, bodyProducer: Optional[IBodyProducer]=None) -> Deferred[IResponse]:
        """
        Issue a new request to the wrapped L{Agent}.

        Send a I{Cookie} header if a cookie for C{uri} is stored in
        L{CookieAgent.cookieJar}. Cookies are automatically extracted and
        stored from requests.

        If a C{'cookie'} header appears in C{headers} it will override the
        automatic cookie header obtained from the cookie jar.

        @see: L{Agent.request}
        """
        actualHeaders = headers if headers is not None else Headers()
        lastRequest = _FakeStdlibRequest(uri)
        if not actualHeaders.hasHeader(b'cookie'):
            self.cookieJar.add_cookie_header(lastRequest)
            cookieHeader = lastRequest.get_header('Cookie', None)
            if cookieHeader is not None:
                actualHeaders = actualHeaders.copy()
                actualHeaders.addRawHeader(b'cookie', networkString(cookieHeader))
        return self._agent.request(method, uri, actualHeaders, bodyProducer).addCallback(self._extractCookies, lastRequest)

    def _extractCookies(self, response: IResponse, request: _FakeStdlibRequest) -> IResponse:
        """
        Extract response cookies and store them in the cookie jar.

        @param response: the Twisted Web response that we are processing.

        @param request: A L{_FakeStdlibRequest} wrapping our Twisted request,
            for L{CookieJar} to extract cookies from.
        """
        self.cookieJar.extract_cookies(_FakeStdlibResponse(response), request)
        return response