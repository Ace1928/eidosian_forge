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
class CombinedLogFormatterTests(unittest.TestCase):
    """
    Tests for L{twisted.web.http.combinedLogFormatter}.
    """

    def test_interface(self):
        """
        L{combinedLogFormatter} provides L{IAccessLogFormatter}.
        """
        self.assertTrue(verifyObject(iweb.IAccessLogFormatter, http.combinedLogFormatter))

    def test_nonASCII(self):
        """
        Bytes in fields of the request which are not part of ASCII are escaped
        in the result.
        """
        reactor = Clock()
        reactor.advance(1234567890)
        timestamp = http.datetimeToLogString(reactor.seconds())
        request = DummyRequestForLogTest(http.HTTPFactory(reactor=reactor))
        request.client = IPv4Address('TCP', b'evil x-forwarded-for \x80', 12345)
        request.method = b'POS\x81'
        request.protocol = b'HTTP/1.\x82'
        request.requestHeaders.addRawHeader(b'referer', b'evil \x83')
        request.requestHeaders.addRawHeader(b'user-agent', b'evil \x84')
        line = http.combinedLogFormatter(timestamp, request)
        self.assertEqual('"evil x-forwarded-for \\x80" - - [13/Feb/2009:23:31:30 +0000] "POS\\x81 /dummy HTTP/1.0" 123 - "evil \\x83" "evil \\x84"', line)

    def test_clientAddrIPv6(self):
        """
        A request from an IPv6 client is logged with that IP address.
        """
        reactor = Clock()
        reactor.advance(1234567890)
        timestamp = http.datetimeToLogString(reactor.seconds())
        request = DummyRequestForLogTest(http.HTTPFactory(reactor=reactor))
        request.client = IPv6Address('TCP', b'::1', 12345)
        line = http.combinedLogFormatter(timestamp, request)
        self.assertEqual('"::1" - - [13/Feb/2009:23:31:30 +0000] "GET /dummy HTTP/1.0" 123 - "-" "-"', line)

    def test_clientAddrUnknown(self):
        """
        A request made from an unknown address type is logged as C{"-"}.
        """

        @implementer(interfaces.IAddress)
        class UnknowableAddress:
            """
            An L{IAddress} which L{combinedLogFormatter} cannot have
            foreknowledge of.
            """
        reactor = Clock()
        reactor.advance(1234567890)
        timestamp = http.datetimeToLogString(reactor.seconds())
        request = DummyRequestForLogTest(http.HTTPFactory(reactor=reactor))
        request.client = UnknowableAddress()
        line = http.combinedLogFormatter(timestamp, request)
        self.assertTrue(line.startswith('"-" '))