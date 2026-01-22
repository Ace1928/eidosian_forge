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
class ContentDecoderAgent:
    """
    An L{Agent} wrapper to handle encoded content.

    It takes care of declaring the support for content in the
    I{Accept-Encoding} header and automatically decompresses the received data
    if the I{Content-Encoding} header indicates a supported encoding.

    For example::

        agent = ContentDecoderAgent(Agent(reactor),
                                    [(b'gzip', GzipDecoder)])

    @param agent: The agent to wrap
    @type agent: L{IAgent}

    @param decoders: A sequence of (name, decoder) objects. The name
        declares which encoding the decoder supports. The decoder must accept
        an L{IResponse} and return an L{IResponse} when called. The order
        determines how the decoders are advertised to the server. Names must
        be unique.not be duplicated.
    @type decoders: sequence of (L{bytes}, L{callable}) tuples

    @since: 11.1

    @see: L{GzipDecoder}
    """

    def __init__(self, agent, decoders):
        self._agent = agent
        self._decoders = dict(decoders)
        self._supported = b','.join([decoder[0] for decoder in decoders])

    def request(self, method, uri, headers=None, bodyProducer=None):
        """
        Send a client request which declares supporting compressed content.

        @see: L{Agent.request}.
        """
        if headers is None:
            headers = Headers()
        else:
            headers = headers.copy()
        headers.addRawHeader(b'accept-encoding', self._supported)
        deferred = self._agent.request(method, uri, headers, bodyProducer)
        return deferred.addCallback(self._handleResponse)

    def _handleResponse(self, response):
        """
        Check if the response is encoded, and wrap it to handle decompression.
        """
        contentEncodingHeaders = response.headers.getRawHeaders(b'content-encoding', [])
        contentEncodingHeaders = b','.join(contentEncodingHeaders).split(b',')
        while contentEncodingHeaders:
            name = contentEncodingHeaders.pop().strip()
            decoder = self._decoders.get(name)
            if decoder is not None:
                response = decoder(response)
            else:
                contentEncodingHeaders.append(name)
                break
        if contentEncodingHeaders:
            response.headers.setRawHeaders(b'content-encoding', [b','.join(contentEncodingHeaders)])
        else:
            response.headers.removeHeader(b'content-encoding')
        return response