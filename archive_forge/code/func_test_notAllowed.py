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
def test_notAllowed(self):
    """
        When an unsupported method is requested, the default
        L{_computeAllowedMethods} method will be called to determine the
        allowed methods, and the HTTP 405 'Method Not Allowed' status will
        be returned with the allowed methods will be returned in the
        'Allow' header.
        """
    req = self._getReq()
    req.requestReceived(b'POST', b'/gettableresource', b'HTTP/1.0')
    self.assertEqual(req.code, 405)
    self.assertEqual(set(req.responseHeaders.getRawHeaders(b'allow')[0].split(b', ')), {b'GET', b'HEAD', b'fred_render_ethel'})