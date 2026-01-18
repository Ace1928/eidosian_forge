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
def test_notAllowedMethod(self):
    """
        When trying to invoke a method not in the allowed method list, we get
        a response saying it is not allowed.
        """
    req = self._getReq()
    req.requestReceived(b'POST', b'/newrender', b'HTTP/1.0')
    self.assertEqual(req.code, 405)
    self.assertTrue(req.responseHeaders.hasHeader(b'allow'))
    raw_header = req.responseHeaders.getRawHeaders(b'allow')[0]
    allowed = sorted((h.strip() for h in raw_header.split(b',')))
    self.assertEqual([b'GET', b'HEAD', b'HEH'], allowed)