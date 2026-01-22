import re
from time import time
from typing import Optional, Tuple
from urllib.parse import ParseResult, urldefrag, urlparse, urlunparse
from twisted.internet import defer
from twisted.internet.protocol import ClientFactory
from twisted.web.http import HTTPClient
from scrapy import Request
from scrapy.http import Headers
from scrapy.responsetypes import responsetypes
from scrapy.utils.httpobj import urlparse_cached
from scrapy.utils.python import to_bytes, to_unicode
class ScrapyHTTPClientFactory(ClientFactory):
    protocol = ScrapyHTTPPageGetter
    waiting = 1
    noisy = False
    followRedirect = False
    afterFoundGet = False

    def _build_response(self, body, request):
        request.meta['download_latency'] = self.headers_time - self.start_time
        status = int(self.status)
        headers = Headers(self.response_headers)
        respcls = responsetypes.from_args(headers=headers, url=self._url, body=body)
        return respcls(url=self._url, status=status, headers=headers, body=body, protocol=to_unicode(self.version))

    def _set_connection_attributes(self, request):
        parsed = urlparse_cached(request)
        self.scheme, self.netloc, self.host, self.port, self.path = _parsed_url_args(parsed)
        proxy = request.meta.get('proxy')
        if proxy:
            self.scheme, _, self.host, self.port, _ = _parse(proxy)
            self.path = self.url

    def __init__(self, request: Request, timeout: float=180):
        self._url: str = urldefrag(request.url)[0]
        self.url: bytes = to_bytes(self._url, encoding='ascii')
        self.method: bytes = to_bytes(request.method, encoding='ascii')
        self.body: Optional[bytes] = request.body or None
        self.headers: Headers = Headers(request.headers)
        self.response_headers: Optional[Headers] = None
        self.timeout: float = request.meta.get('download_timeout') or timeout
        self.start_time: float = time()
        self.deferred: defer.Deferred = defer.Deferred().addCallback(self._build_response, request)
        self._disconnectedDeferred: defer.Deferred = defer.Deferred()
        self._set_connection_attributes(request)
        self.headers.setdefault('Host', self.netloc)
        if self.body is not None:
            self.headers['Content-Length'] = len(self.body)
            self.headers.setdefault('Connection', 'close')
        elif self.method == b'POST':
            self.headers['Content-Length'] = 0

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__}: {self._url}>'

    def _cancelTimeout(self, result, timeoutCall):
        if timeoutCall.active():
            timeoutCall.cancel()
        return result

    def buildProtocol(self, addr):
        p = ClientFactory.buildProtocol(self, addr)
        p.followRedirect = self.followRedirect
        p.afterFoundGet = self.afterFoundGet
        if self.timeout:
            from twisted.internet import reactor
            timeoutCall = reactor.callLater(self.timeout, p.timeout)
            self.deferred.addBoth(self._cancelTimeout, timeoutCall)
        return p

    def gotHeaders(self, headers):
        self.headers_time = time()
        self.response_headers = headers

    def gotStatus(self, version, status, message):
        """
        Set the status of the request on us.
        @param version: The HTTP version.
        @type version: L{bytes}
        @param status: The HTTP status code, an integer represented as a
        bytestring.
        @type status: L{bytes}
        @param message: The HTTP status message.
        @type message: L{bytes}
        """
        self.version, self.status, self.message = (version, status, message)

    def page(self, page):
        if self.waiting:
            self.waiting = 0
            self.deferred.callback(page)

    def noPage(self, reason):
        if self.waiting:
            self.waiting = 0
            self.deferred.errback(reason)

    def clientConnectionFailed(self, _, reason):
        """
        When a connection attempt fails, the request cannot be issued.  If no
        result has yet been provided to the result Deferred, provide the
        connection failure reason as an error result.
        """
        if self.waiting:
            self.waiting = 0
            self._disconnectedDeferred.callback(None)
            self.deferred.errback(reason)