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
def test_sessionAttribute(self):
    """
        On a L{Request}, the C{session} attribute retrieves the associated
        L{Session} only if it has been initialized.  If the request is secure,
        it retrieves the secure session.
        """
    site = server.Site(resource.Resource())
    d = DummyChannel()
    d.transport = DummyChannel.SSL()
    request = server.Request(d, 1)
    request.site = site
    request.sitepath = []
    self.assertIs(request.session, None)
    insecureSession = request.getSession(forceNotSecure=True)
    self.addCleanup(insecureSession.expire)
    self.assertIs(request.session, None)
    secureSession = request.getSession()
    self.addCleanup(secureSession.expire)
    self.assertIsNot(secureSession, None)
    self.assertIsNot(secureSession, insecureSession)
    self.assertIs(request.session, secureSession)