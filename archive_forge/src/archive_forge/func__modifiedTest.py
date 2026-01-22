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