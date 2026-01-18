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