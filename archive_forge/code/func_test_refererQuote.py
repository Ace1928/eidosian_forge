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
def test_refererQuote(self):
    """
        If the value of the I{Referer} header contains a quote, the quote is
        escaped.
        """
    self.site._logDateTime = '[%02d/%3s/%4d:%02d:%02d:%02d +0000]' % (25, 'Oct', 2004, 12, 31, 59)
    self.request.requestHeaders.addRawHeader(b'referer', b'http://malicious" ".website.invalid')
    self.assertLogs(b'"1.2.3.4" - - [25/Oct/2004:12:31:59 +0000] "GET /dummy HTTP/1.0" 123 - "http://malicious\\" \\".website.invalid" "-"\n')