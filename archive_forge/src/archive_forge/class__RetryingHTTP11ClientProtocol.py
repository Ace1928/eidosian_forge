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
class _RetryingHTTP11ClientProtocol:
    """
    A wrapper for L{HTTP11ClientProtocol} that automatically retries requests.

    @ivar _clientProtocol: The underlying L{HTTP11ClientProtocol}.

    @ivar _newConnection: A callable that creates a new connection for a
        retry.
    """

    def __init__(self, clientProtocol, newConnection):
        self._clientProtocol = clientProtocol
        self._newConnection = newConnection

    def _shouldRetry(self, method, exception, bodyProducer):
        """
        Indicate whether request should be retried.

        Only returns C{True} if method is idempotent, no response was
        received, the reason for the failed request was not due to
        user-requested cancellation, and no body was sent. The latter
        requirement may be relaxed in the future, and PUT added to approved
        method list.

        @param method: The method of the request.
        @type method: L{bytes}
        """
        if method not in (b'GET', b'HEAD', b'OPTIONS', b'DELETE', b'TRACE'):
            return False
        if not isinstance(exception, (RequestNotSent, RequestTransmissionFailed, ResponseNeverReceived)):
            return False
        if isinstance(exception, _WrapperException):
            for aFailure in exception.reasons:
                if aFailure.check(defer.CancelledError):
                    return False
        if bodyProducer is not None:
            return False
        return True

    def request(self, request):
        """
        Do a request, and retry once (with a new connection) if it fails in
        a retryable manner.

        @param request: A L{Request} instance that will be requested using the
            wrapped protocol.
        """
        d = self._clientProtocol.request(request)

        def failed(reason):
            if self._shouldRetry(request.method, reason.value, request.bodyProducer):
                return self._newConnection().addCallback(lambda connection: connection.request(request))
            else:
                return reason
        d.addErrback(failed)
        return d