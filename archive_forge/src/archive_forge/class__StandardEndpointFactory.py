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
@implementer(IAgentEndpointFactory)
class _StandardEndpointFactory:
    """
    Standard HTTP endpoint destinations - TCP for HTTP, TCP+TLS for HTTPS.

    @ivar _policyForHTTPS: A web context factory which will be used to create
        SSL context objects for any SSL connections the agent needs to make.

    @ivar _connectTimeout: If not L{None}, the timeout passed to
        L{HostnameEndpoint} for specifying the connection timeout.

    @ivar _bindAddress: If not L{None}, the address passed to
        L{HostnameEndpoint} for specifying the local address to bind to.
    """

    def __init__(self, reactor, contextFactory, connectTimeout, bindAddress):
        """
        @param reactor: A provider to use to create endpoints.
        @type reactor: see L{HostnameEndpoint.__init__} for acceptable reactor
            types.

        @param contextFactory: A factory for TLS contexts, to control the
            verification parameters of OpenSSL.
        @type contextFactory: L{IPolicyForHTTPS}.

        @param connectTimeout: The amount of time that this L{Agent} will wait
            for the peer to accept a connection.
        @type connectTimeout: L{float} or L{None}

        @param bindAddress: The local address for client sockets to bind to.
        @type bindAddress: L{bytes} or L{None}
        """
        self._reactor = reactor
        self._policyForHTTPS = contextFactory
        self._connectTimeout = connectTimeout
        self._bindAddress = bindAddress

    def endpointForURI(self, uri):
        """
        Connect directly over TCP for C{b'http'} scheme, and TLS for
        C{b'https'}.

        @param uri: L{URI} to connect to.

        @return: Endpoint to connect to.
        @rtype: L{IStreamClientEndpoint}
        """
        kwargs = {}
        if self._connectTimeout is not None:
            kwargs['timeout'] = self._connectTimeout
        kwargs['bindAddress'] = self._bindAddress
        try:
            host = nativeString(uri.host)
        except UnicodeDecodeError:
            raise ValueError('The host of the provided URI ({uri.host!r}) contains non-ASCII octets, it should be ASCII decodable.'.format(uri=uri))
        endpoint = HostnameEndpoint(self._reactor, host, uri.port, **kwargs)
        if uri.scheme == b'http':
            return endpoint
        elif uri.scheme == b'https':
            connectionCreator = self._policyForHTTPS.creatorForNetloc(uri.host, uri.port)
            return wrapClientTLS(connectionCreator, endpoint)
        else:
            raise SchemeNotSupported(f'Unsupported scheme: {uri.scheme!r}')