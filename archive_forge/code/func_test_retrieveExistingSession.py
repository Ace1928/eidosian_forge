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
def test_retrieveExistingSession(self):
    """
        L{Request.getSession} retrieves an existing session if the relevant
        cookie is set in the incoming request.
        """
    site = server.Site(resource.Resource())
    d = DummyChannel()
    request = server.Request(d, 1)
    request.site = site
    request.sitepath = []
    mySession = server.Session(site, b'special-id')
    site.sessions[mySession.uid] = mySession
    request.received_cookies[b'TWISTED_SESSION'] = mySession.uid
    self.assertIs(request.getSession(), mySession)