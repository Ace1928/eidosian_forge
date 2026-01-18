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
def test_sessionCaching(self):
    """
        L{Request.getSession} creates the session object only once per request;
        if it is called twice it returns the identical result.
        """
    site = server.Site(resource.Resource())
    d = DummyChannel()
    request = server.Request(d, 1)
    request.site = site
    request.sitepath = []
    session1 = request.getSession()
    self.addCleanup(session1.expire)
    session2 = request.getSession()
    self.assertIs(session1, session2)