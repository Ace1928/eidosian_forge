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
def test_siteSuppliedContentFile(self):
    """
        L{http.Request} uses L{Site.getContentFile}, if it exists, to get a
        file-like object for the request content.
        """
    lengths = []
    contentFile = BytesIO()
    site = server.Site(resource.Resource())

    def getContentFile(length):
        lengths.append(length)
        return contentFile
    site.getContentFile = getContentFile
    channel = DummyChannel()
    channel.site = site
    request = server.Request(channel)
    request.gotLength(12345)
    self.assertEqual([12345], lengths)
    self.assertIs(contentFile, request.content)