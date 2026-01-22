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
class LogEscapingTests(unittest.TestCase):

    def setUp(self):
        self.logPath = self.mktemp()
        self.site = http.HTTPFactory(self.logPath)
        self.site.startFactory()
        self.request = DummyRequestForLogTest(self.site, False)

    def assertLogs(self, line):
        """
        Assert that if C{self.request} is logged using C{self.site} then
        C{line} is written to the site's access log file.

        @param line: The expected line.
        @type line: L{bytes}

        @raise self.failureException: If the log file contains something other
            than the expected line.
        """
        try:
            self.site.log(self.request)
        finally:
            self.site.stopFactory()
        logged = FilePath(self.logPath).getContent()
        self.assertEqual(line, logged)

    def test_simple(self):
        """
        A I{GET} request is logged with no extra escapes.
        """
        self.site._logDateTime = '[%02d/%3s/%4d:%02d:%02d:%02d +0000]' % (25, 'Oct', 2004, 12, 31, 59)
        self.assertLogs(b'"1.2.3.4" - - [25/Oct/2004:12:31:59 +0000] "GET /dummy HTTP/1.0" 123 - "-" "-"\n')

    def test_methodQuote(self):
        """
        If the HTTP request method includes a quote, the quote is escaped.
        """
        self.site._logDateTime = '[%02d/%3s/%4d:%02d:%02d:%02d +0000]' % (25, 'Oct', 2004, 12, 31, 59)
        self.request.method = b'G"T'
        self.assertLogs(b'"1.2.3.4" - - [25/Oct/2004:12:31:59 +0000] "G\\"T /dummy HTTP/1.0" 123 - "-" "-"\n')

    def test_requestQuote(self):
        """
        If the HTTP request path includes a quote, the quote is escaped.
        """
        self.site._logDateTime = '[%02d/%3s/%4d:%02d:%02d:%02d +0000]' % (25, 'Oct', 2004, 12, 31, 59)
        self.request.uri = b'/dummy"withquote'
        self.assertLogs(b'"1.2.3.4" - - [25/Oct/2004:12:31:59 +0000] "GET /dummy\\"withquote HTTP/1.0" 123 - "-" "-"\n')

    def test_protoQuote(self):
        """
        If the HTTP request version includes a quote, the quote is escaped.
        """
        self.site._logDateTime = '[%02d/%3s/%4d:%02d:%02d:%02d +0000]' % (25, 'Oct', 2004, 12, 31, 59)
        self.request.clientproto = b'HT"P/1.0'
        self.assertLogs(b'"1.2.3.4" - - [25/Oct/2004:12:31:59 +0000] "GET /dummy HT\\"P/1.0" 123 - "-" "-"\n')

    def test_refererQuote(self):
        """
        If the value of the I{Referer} header contains a quote, the quote is
        escaped.
        """
        self.site._logDateTime = '[%02d/%3s/%4d:%02d:%02d:%02d +0000]' % (25, 'Oct', 2004, 12, 31, 59)
        self.request.requestHeaders.addRawHeader(b'referer', b'http://malicious" ".website.invalid')
        self.assertLogs(b'"1.2.3.4" - - [25/Oct/2004:12:31:59 +0000] "GET /dummy HTTP/1.0" 123 - "http://malicious\\" \\".website.invalid" "-"\n')

    def test_userAgentQuote(self):
        """
        If the value of the I{User-Agent} header contains a quote, the quote is
        escaped.
        """
        self.site._logDateTime = '[%02d/%3s/%4d:%02d:%02d:%02d +0000]' % (25, 'Oct', 2004, 12, 31, 59)
        self.request.requestHeaders.addRawHeader(b'user-agent', b'Malicious Web" Evil')
        self.assertLogs(b'"1.2.3.4" - - [25/Oct/2004:12:31:59 +0000] "GET /dummy HTTP/1.0" 123 - "-" "Malicious Web\\" Evil"\n')