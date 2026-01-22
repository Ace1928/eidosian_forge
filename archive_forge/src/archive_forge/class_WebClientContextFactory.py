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
class WebClientContextFactory:
    """
    This class is deprecated.  Please simply use L{Agent} as-is, or if you want
    to customize something, use L{BrowserLikePolicyForHTTPS}.

    A L{WebClientContextFactory} is an HTTPS policy which totally ignores the
    hostname and port.  It performs basic certificate verification, however the
    lack of validation of service identity (e.g.  hostname validation) means it
    is still vulnerable to man-in-the-middle attacks.  Don't use it any more.
    """

    def _getCertificateOptions(self, hostname, port):
        """
        Return a L{CertificateOptions}.

        @param hostname: ignored

        @param port: ignored

        @return: A new CertificateOptions instance.
        @rtype: L{CertificateOptions}
        """
        return CertificateOptions(method=SSL.SSLv23_METHOD, trustRoot=platformTrust())

    @_requireSSL
    def getContext(self, hostname, port):
        """
        Return an L{OpenSSL.SSL.Context}.

        @param hostname: ignored
        @param port: ignored

        @return: A new SSL context.
        @rtype: L{OpenSSL.SSL.Context}
        """
        return self._getCertificateOptions(hostname, port).getContext()