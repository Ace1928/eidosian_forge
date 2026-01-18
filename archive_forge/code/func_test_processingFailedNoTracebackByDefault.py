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
def test_processingFailedNoTracebackByDefault(self):
    """
        By default, L{Request.processingFailed} does not write out the failure,
        but give a generic error message, as L{Site.displayTracebacks} is
        disabled by default.
        """
    logObserver = EventLoggingObserver.createWithCleanup(self, globalLogPublisher)
    d = DummyChannel()
    request = server.Request(d, 1)
    request.site = server.Site(resource.Resource())
    fail = failure.Failure(Exception('Oh no!'))
    request.processingFailed(fail)
    self.assertNotIn(b'Oh no!', request.transport.written.getvalue())
    self.assertIn(b'Processing Failed', request.transport.written.getvalue())
    self.assertEquals(1, len(logObserver))
    event = logObserver[0]
    f = event['log_failure']
    self.assertIsInstance(f.value, Exception)
    self.assertEquals(f.getErrorMessage(), 'Oh no!')
    self.assertEqual(1, len(self.flushLoggedErrors()))