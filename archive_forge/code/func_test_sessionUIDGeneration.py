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
def test_sessionUIDGeneration(self):
    """
        L{site.getSession} generates L{Session} objects with distinct UIDs from
        a secure source of entropy.
        """
    site = server.Site(resource.Resource())
    self.assertIdentical(site._entropy, os.urandom)

    def predictableEntropy(n):
        predictableEntropy.x += 1
        return (chr(predictableEntropy.x) * n).encode('charmap')
    predictableEntropy.x = 0
    self.patch(site, '_entropy', predictableEntropy)
    a = self.getAutoExpiringSession(site)
    b = self.getAutoExpiringSession(site)
    self.assertEqual(a.uid, b'01' * 32)
    self.assertEqual(b.uid, b'02' * 32)
    self.assertEqual(site.counter, 2)