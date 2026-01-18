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
@classmethod
def usingEndpointFactory(cls, reactor, endpointFactory, pool=None):
    """
        Create a new L{Agent} that will use the endpoint factory to figure
        out how to connect to the server.

        @param reactor: A reactor for this L{Agent} to place outgoing
            connections.
        @type reactor: see L{HostnameEndpoint.__init__} for acceptable reactor
            types.

        @param endpointFactory: Used to construct endpoints which the
            HTTP client will connect with.
        @type endpointFactory: an L{IAgentEndpointFactory} provider.

        @param pool: An L{HTTPConnectionPool} instance, or L{None}, in which
            case a non-persistent L{HTTPConnectionPool} instance will be
            created.
        @type pool: L{HTTPConnectionPool}

        @return: A new L{Agent}.
        """
    agent = cls.__new__(cls)
    agent._init(reactor, endpointFactory, pool)
    return agent