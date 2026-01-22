import base64
import calendar
import random
from io import BytesIO
from itertools import cycle
from typing import Sequence, Union
from unittest import skipIf
from urllib.parse import clear_cache  # type: ignore[attr-defined]
from urllib.parse import urlparse, urlunsplit
from zope.interface import directlyProvides, providedBy, provider
from zope.interface.verify import verifyObject
import hamcrest
from twisted.internet import address
from twisted.internet.error import ConnectionDone, ConnectionLost
from twisted.internet.task import Clock
from twisted.internet.testing import (
from twisted.logger import globalLogPublisher
from twisted.protocols import loopback
from twisted.python.compat import iterbytes, networkString
from twisted.python.components import proxyForInterface
from twisted.python.failure import Failure
from twisted.test.test_internet import DummyProducer
from twisted.trial import unittest
from twisted.trial.unittest import TestCase
from twisted.web import http, http_headers, iweb
from twisted.web.http import PotentialDataLoss, _DataLoss, _IdentityTransferDecoder
from twisted.web.test.requesthelper import (
from ._util import assertIsFilesystemTemporary
class MultilineHeadersTests(unittest.TestCase):
    """
    Tests to exercise handling of multiline headers by L{HTTPClient}.  RFCs 1945
    (HTTP 1.0) and 2616 (HTTP 1.1) state that HTTP message header fields can
    span multiple lines if each extra line is preceded by at least one space or
    horizontal tab.
    """

    def setUp(self):
        """
        Initialize variables used to verify that the header-processing functions
        are getting called.
        """
        self.handleHeaderCalled = False
        self.handleEndHeadersCalled = False
    expectedHeaders = {b'Content-Length': b'10', b'X-Multiline': b'line-0\tline-1', b'X-Multiline2': b'line-2 line-3'}

    def ourHandleHeader(self, key, val):
        """
        Dummy implementation of L{HTTPClient.handleHeader}.
        """
        self.handleHeaderCalled = True
        self.assertEqual(val, self.expectedHeaders[key])

    def ourHandleEndHeaders(self):
        """
        Dummy implementation of L{HTTPClient.handleEndHeaders}.
        """
        self.handleEndHeadersCalled = True

    def test_extractHeader(self):
        """
        A header isn't processed by L{HTTPClient.extractHeader} until it is
        confirmed in L{HTTPClient.lineReceived} that the header has been
        received completely.
        """
        c = ClientDriver()
        c.handleHeader = self.ourHandleHeader
        c.handleEndHeaders = self.ourHandleEndHeaders
        c.lineReceived(b'HTTP/1.0 201')
        c.lineReceived(b'Content-Length: 10')
        self.assertIdentical(c.length, None)
        self.assertFalse(self.handleHeaderCalled)
        self.assertFalse(self.handleEndHeadersCalled)
        c.lineReceived(b'')
        self.assertTrue(self.handleHeaderCalled)
        self.assertTrue(self.handleEndHeadersCalled)
        self.assertEqual(c.length, 10)

    def test_noHeaders(self):
        """
        An HTTP request with no headers will not cause any calls to
        L{handleHeader} but will cause L{handleEndHeaders} to be called on
        L{HTTPClient} subclasses.
        """
        c = ClientDriver()
        c.handleHeader = self.ourHandleHeader
        c.handleEndHeaders = self.ourHandleEndHeaders
        c.lineReceived(b'HTTP/1.0 201')
        c.lineReceived(b'')
        self.assertFalse(self.handleHeaderCalled)
        self.assertTrue(self.handleEndHeadersCalled)
        self.assertEqual(c.version, b'HTTP/1.0')
        self.assertEqual(c.status, b'201')

    def test_multilineHeaders(self):
        """
        L{HTTPClient} parses multiline headers by buffering header lines until
        an empty line or a line that does not start with whitespace hits
        lineReceived, confirming that the header has been received completely.
        """
        c = ClientDriver()
        c.handleHeader = self.ourHandleHeader
        c.handleEndHeaders = self.ourHandleEndHeaders
        c.lineReceived(b'HTTP/1.0 201')
        c.lineReceived(b'X-Multiline: line-0')
        self.assertFalse(self.handleHeaderCalled)
        c.lineReceived(b'\tline-1')
        c.lineReceived(b'X-Multiline2: line-2')
        self.assertTrue(self.handleHeaderCalled)
        c.lineReceived(b' line-3')
        c.lineReceived(b'Content-Length: 10')
        c.lineReceived(b'')
        self.assertTrue(self.handleEndHeadersCalled)
        self.assertEqual(c.version, b'HTTP/1.0')
        self.assertEqual(c.status, b'201')
        self.assertEqual(c.length, 10)