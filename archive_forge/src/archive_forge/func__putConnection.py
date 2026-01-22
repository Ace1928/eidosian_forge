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
def _putConnection(self, key, connection):
    """
        Return a persistent connection to the pool. This will be called by
        L{HTTP11ClientProtocol} when the connection becomes quiescent.
        """
    if connection.state != 'QUIESCENT':
        try:
            raise RuntimeError('BUG: Non-quiescent protocol added to connection pool.')
        except BaseException:
            self._log.failure('BUG: Non-quiescent protocol added to connection pool.')
        return
    connections = self._connections.setdefault(key, [])
    if len(connections) == self.maxPersistentPerHost:
        dropped = connections.pop(0)
        dropped.transport.loseConnection()
        self._timeouts[dropped].cancel()
        del self._timeouts[dropped]
    connections.append(connection)
    cid = self._reactor.callLater(self.cachedConnectionTimeout, self._removeConnection, key, connection)
    self._timeouts[connection] = cid