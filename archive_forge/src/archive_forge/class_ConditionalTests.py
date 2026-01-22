import os
import zlib
from io import BytesIO
from typing import List
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.internet import interfaces
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.task import Clock
from twisted.internet.testing import EventLoggingObserver, StringTransport
from twisted.logger import LogLevel, globalLogPublisher
from twisted.python import failure, reflect
from twisted.python.compat import iterbytes
from twisted.python.filepath import FilePath
from twisted.trial import unittest
from twisted.web import error, http, iweb, resource, server
from twisted.web.resource import Resource
from twisted.web.server import NOT_DONE_YET, Request, Site
from twisted.web.static import Data
from twisted.web.test.requesthelper import DummyChannel, DummyRequest
from ._util import assertIsFilesystemTemporary
class ConditionalTests(unittest.TestCase):
    """
    web.server's handling of conditional requests for cache validation.
    """

    def setUp(self):
        self.resrc = SimpleResource()
        self.resrc.putChild(b'', self.resrc)
        self.resrc.putChild(b'with-content-type', SimpleResource(b'image/jpeg'))
        self.site = server.Site(self.resrc)
        self.site.startFactory()
        self.addCleanup(self.site.stopFactory)
        self.channel = self.site.buildProtocol(None)
        self.transport = http.StringTransport()
        self.transport.close = lambda *a, **kw: None
        self.transport.disconnecting = lambda *a, **kw: 0
        self.transport.getPeer = lambda *a, **kw: 'peer'
        self.transport.getHost = lambda *a, **kw: 'host'
        self.channel.makeConnection(self.transport)

    def tearDown(self):
        self.channel.connectionLost(None)

    def _modifiedTest(self, modifiedSince=None, etag=None):
        """
        Given the value C{modifiedSince} for the I{If-Modified-Since} header or
        the value C{etag} for the I{If-Not-Match} header, verify that a response
        with a 200 code, a default Content-Type, and the resource as the body is
        returned.
        """
        if modifiedSince is not None:
            validator = b'If-Modified-Since: ' + modifiedSince
        else:
            validator = b'If-Not-Match: ' + etag
        for line in [b'GET / HTTP/1.1', validator, b'']:
            self.channel.dataReceived(line + b'\r\n')
        result = self.transport.getvalue()
        self.assertEqual(httpCode(result), http.OK)
        self.assertEqual(httpBody(result), b'correct')
        self.assertEqual(httpHeader(result, b'Content-Type'), b'text/html')

    def test_modified(self):
        """
        If a request is made with an I{If-Modified-Since} header value with
        a timestamp indicating a time before the last modification of the
        requested resource, a 200 response is returned along with a response
        body containing the resource.
        """
        self._modifiedTest(modifiedSince=http.datetimeToString(1))

    def test_unmodified(self):
        """
        If a request is made with an I{If-Modified-Since} header value with a
        timestamp indicating a time after the last modification of the request
        resource, a 304 response is returned along with an empty response body
        and no Content-Type header if the application does not set one.
        """
        for line in [b'GET / HTTP/1.1', b'If-Modified-Since: ' + http.datetimeToString(100), b'']:
            self.channel.dataReceived(line + b'\r\n')
        result = self.transport.getvalue()
        self.assertEqual(httpCode(result), http.NOT_MODIFIED)
        self.assertEqual(httpBody(result), b'')
        self.assertEqual(httpHeader(result, b'Content-Type'), None)

    def test_invalidTimestamp(self):
        """
        If a request is made with an I{If-Modified-Since} header value which
        cannot be parsed, the header is treated as not having been present
        and a normal 200 response is returned with a response body
        containing the resource.
        """
        self._modifiedTest(modifiedSince=b'like, maybe a week ago, I guess?')

    def test_invalidTimestampYear(self):
        """
        If a request is made with an I{If-Modified-Since} header value which
        contains a string in the year position which is not an integer, the
        header is treated as not having been present and a normal 200
        response is returned with a response body containing the resource.
        """
        self._modifiedTest(modifiedSince=b'Thu, 01 Jan blah 00:00:10 GMT')

    def test_invalidTimestampTooLongAgo(self):
        """
        If a request is made with an I{If-Modified-Since} header value which
        contains a year before the epoch, the header is treated as not
        having been present and a normal 200 response is returned with a
        response body containing the resource.
        """
        self._modifiedTest(modifiedSince=b'Thu, 01 Jan 1899 00:00:10 GMT')

    def test_invalidTimestampMonth(self):
        """
        If a request is made with an I{If-Modified-Since} header value which
        contains a string in the month position which is not a recognized
        month abbreviation, the header is treated as not having been present
        and a normal 200 response is returned with a response body
        containing the resource.
        """
        self._modifiedTest(modifiedSince=b'Thu, 01 Blah 1970 00:00:10 GMT')

    def test_etagMatchedNot(self):
        """
        If a request is made with an I{If-None-Match} ETag which does not match
        the current ETag of the requested resource, the header is treated as not
        having been present and a normal 200 response is returned with a
        response body containing the resource.
        """
        self._modifiedTest(etag=b'unmatchedTag')

    def test_etagMatched(self):
        """
        If a request is made with an I{If-None-Match} ETag which does match the
        current ETag of the requested resource, a 304 response is returned along
        with an empty response body.
        """
        for line in [b'GET / HTTP/1.1', b'If-None-Match: MatchingTag', b'']:
            self.channel.dataReceived(line + b'\r\n')
        result = self.transport.getvalue()
        self.assertEqual(httpHeader(result, b'ETag'), b'MatchingTag')
        self.assertEqual(httpCode(result), http.NOT_MODIFIED)
        self.assertEqual(httpBody(result), b'')

    def test_unmodifiedWithContentType(self):
        """
        Similar to L{test_etagMatched}, but the response should include a
        I{Content-Type} header if the application explicitly sets one.

        This I{Content-Type} header SHOULD NOT be present according to RFC 2616,
        section 10.3.5.  It will only be present if the application explicitly
        sets it.
        """
        for line in [b'GET /with-content-type HTTP/1.1', b'If-None-Match: MatchingTag', b'']:
            self.channel.dataReceived(line + b'\r\n')
        result = self.transport.getvalue()
        self.assertEqual(httpCode(result), http.NOT_MODIFIED)
        self.assertEqual(httpBody(result), b'')
        self.assertEqual(httpHeader(result, b'Content-Type'), b'image/jpeg')