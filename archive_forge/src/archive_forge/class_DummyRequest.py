from __future__ import annotations
from io import BytesIO
from typing import Dict, List, Optional
from zope.interface import implementer, verify
from incremental import Version
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IAddress, ISSLTransport
from twisted.internet.task import Clock
from twisted.python.deprecate import deprecated
from twisted.trial import unittest
from twisted.web._responses import FOUND
from twisted.web.http_headers import Headers
from twisted.web.resource import Resource
from twisted.web.server import NOT_DONE_YET, Session, Site
class DummyRequest:
    """
    Represents a dummy or fake request. See L{twisted.web.server.Request}.

    @ivar _finishedDeferreds: L{None} or a C{list} of L{Deferreds} which will
        be called back with L{None} when C{finish} is called or which will be
        errbacked if C{processingFailed} is called.

    @type requestheaders: C{Headers}
    @ivar requestheaders: A Headers instance that stores values for all request
        headers.

    @type responseHeaders: C{Headers}
    @ivar responseHeaders: A Headers instance that stores values for all
        response headers.

    @type responseCode: C{int}
    @ivar responseCode: The response code which was passed to
        C{setResponseCode}.

    @type written: C{list} of C{bytes}
    @ivar written: The bytes which have been written to the request.
    """
    uri = b'http://dummy/'
    method = b'GET'
    client: Optional[IAddress] = None
    sitepath: List[bytes]
    written: List[bytes]
    prepath: List[bytes]
    args: Dict[bytes, List[bytes]]
    _finishedDeferreds: List[Deferred[None]]

    def registerProducer(self, prod, s):
        """
        Call an L{IPullProducer}'s C{resumeProducing} method in a
        loop until it unregisters itself.

        @param prod: The producer.
        @type prod: L{IPullProducer}

        @param s: Whether or not the producer is streaming.
        """
        self.go = 1
        while self.go:
            prod.resumeProducing()

    def unregisterProducer(self):
        self.go = 0

    def __init__(self, postpath: list[bytes], session: Optional[Session]=None, client: Optional[IAddress]=None) -> None:
        self.sitepath = []
        self.written = []
        self.finished = 0
        self.postpath = postpath
        self.prepath = []
        self.session = None
        self.protoSession = session or Session(site=None, uid=b'0', reactor=Clock())
        self.args = {}
        self.requestHeaders = Headers()
        self.responseHeaders = Headers()
        self.responseCode = None
        self._finishedDeferreds = []
        self._serverName = b'dummy'
        self.clientproto = b'HTTP/1.0'

    def getAllHeaders(self):
        """
        Return dictionary mapping the names of all received headers to the last
        value received for each.

        Since this method does not return all header information,
        C{self.requestHeaders.getAllRawHeaders()} may be preferred.

        NOTE: This function is a direct copy of
        C{twisted.web.http.Request.getAllRawHeaders}.
        """
        headers = {}
        for k, v in self.requestHeaders.getAllRawHeaders():
            headers[k.lower()] = v[-1]
        return headers

    def getHeader(self, name):
        """
        Retrieve the value of a request header.

        @type name: C{bytes}
        @param name: The name of the request header for which to retrieve the
            value.  Header names are compared case-insensitively.

        @rtype: C{bytes} or L{None}
        @return: The value of the specified request header.
        """
        return self.requestHeaders.getRawHeaders(name.lower(), [None])[0]

    def setHeader(self, name, value):
        """TODO: make this assert on write() if the header is content-length"""
        self.responseHeaders.addRawHeader(name, value)

    def getSession(self, sessionInterface=None):
        if self.session:
            return self.session
        assert not self.written, 'Session cannot be requested after data has been written.'
        self.session = self.protoSession
        return self.session

    def render(self, resource):
        """
        Render the given resource as a response to this request.

        This implementation only handles a few of the most common behaviors of
        resources.  It can handle a render method that returns a string or
        C{NOT_DONE_YET}.  It doesn't know anything about the semantics of
        request methods (eg HEAD) nor how to set any particular headers.
        Basically, it's largely broken, but sufficient for some tests at least.
        It should B{not} be expanded to do all the same stuff L{Request} does.
        Instead, L{DummyRequest} should be phased out and L{Request} (or some
        other real code factored in a different way) used.
        """
        result = resource.render(self)
        if result is NOT_DONE_YET:
            return
        self.write(result)
        self.finish()

    def write(self, data):
        if not isinstance(data, bytes):
            raise TypeError('write() only accepts bytes')
        self.written.append(data)

    def notifyFinish(self) -> Deferred[None]:
        """
        Return a L{Deferred} which is called back with L{None} when the request
        is finished.  This will probably only work if you haven't called
        C{finish} yet.
        """
        finished: Deferred[None] = Deferred()
        self._finishedDeferreds.append(finished)
        return finished

    def finish(self):
        """
        Record that the request is finished and callback and L{Deferred}s
        waiting for notification of this.
        """
        self.finished = self.finished + 1
        if self._finishedDeferreds is not None:
            observers = self._finishedDeferreds
            self._finishedDeferreds = None
            for obs in observers:
                obs.callback(None)

    def processingFailed(self, reason):
        """
        Errback and L{Deferreds} waiting for finish notification.
        """
        if self._finishedDeferreds is not None:
            observers = self._finishedDeferreds
            self._finishedDeferreds = None
            for obs in observers:
                obs.errback(reason)

    def addArg(self, name, value):
        self.args[name] = [value]

    def setResponseCode(self, code, message=None):
        """
        Set the HTTP status response code, but takes care that this is called
        before any data is written.
        """
        assert not self.written, 'Response code cannot be set after data hasbeen written: {}.'.format('@@@@'.join(self.written))
        self.responseCode = code
        self.responseMessage = message

    def setLastModified(self, when):
        assert not self.written, 'Last-Modified cannot be set after data has been written: {}.'.format('@@@@'.join(self.written))

    def setETag(self, tag):
        assert not self.written, 'ETag cannot be set after data has been written: {}.'.format('@@@@'.join(self.written))

    @deprecated(Version('Twisted', 18, 4, 0), replacement='getClientAddress')
    def getClientIP(self):
        """
        Return the IPv4 address of the client which made this request, if there
        is one, otherwise L{None}.
        """
        if isinstance(self.client, (IPv4Address, IPv6Address)):
            return self.client.host
        return None

    def getClientAddress(self):
        """
        Return the L{IAddress} of the client that made this request.

        @return: an address.
        @rtype: an L{IAddress} provider.
        """
        if self.client is None:
            return NullAddress()
        return self.client

    def getRequestHostname(self):
        """
        Get a dummy hostname associated to the HTTP request.

        @rtype: C{bytes}
        @returns: a dummy hostname
        """
        return self._serverName

    def getHost(self):
        """
        Get a dummy transport's host.

        @rtype: C{IPv4Address}
        @returns: a dummy transport's host
        """
        return IPv4Address('TCP', '127.0.0.1', 80)

    def setHost(self, host, port, ssl=0):
        """
        Change the host and port the request thinks it's using.

        @type host: C{bytes}
        @param host: The value to which to change the host header.

        @type ssl: C{bool}
        @param ssl: A flag which, if C{True}, indicates that the request is
            considered secure (if C{True}, L{isSecure} will return C{True}).
        """
        self._forceSSL = ssl
        if self.isSecure():
            default = 443
        else:
            default = 80
        if port == default:
            hostHeader = host
        else:
            hostHeader = b'%b:%d' % (host, port)
        self.requestHeaders.addRawHeader(b'host', hostHeader)

    def redirect(self, url):
        """
        Utility function that does a redirect.

        The request should have finish() called after this.
        """
        self.setResponseCode(FOUND)
        self.setHeader(b'location', url)