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
def test_retrieveNonExistentSession(self):
    """
        L{Request.getSession} generates a new session if the session ID
        advertised in the cookie from the incoming request is not found.
        """
    site = server.Site(resource.Resource())
    d = DummyChannel()
    request = server.Request(d, 1)
    request.site = site
    request.sitepath = []
    request.received_cookies[b'TWISTED_SESSION'] = b'does-not-exist'
    session = request.getSession()
    self.assertIsNotNone(session)
    self.addCleanup(session.expire)
    self.assertTrue(request.cookies[0].startswith(b'TWISTED_SESSION='))
    self.assertNotIn(b'does-not-exist', request.cookies[0])