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
class PipeliningBodyTests(unittest.TestCase, ResponseTestMixin):
    """
    Tests that multiple pipelined requests with bodies are correctly buffered.
    """
    requests = b'POST / HTTP/1.1\r\nContent-Length: 10\r\n\r\n0123456789POST / HTTP/1.1\r\nContent-Length: 10\r\n\r\n0123456789'
    expectedResponses = [(b'HTTP/1.1 200 OK', b'Request: /', b'Command: POST', b'Version: HTTP/1.1', b'Content-Length: 21', b"'''\n10\n0123456789'''\n"), (b'HTTP/1.1 200 OK', b'Request: /', b'Command: POST', b'Version: HTTP/1.1', b'Content-Length: 21', b"'''\n10\n0123456789'''\n")]

    def test_noPipelining(self):
        """
        Test that pipelined requests get buffered, not processed in parallel.
        """
        b = StringTransport()
        a = http.HTTPChannel()
        a.requestFactory = DelayedHTTPHandlerProxy
        a.makeConnection(b)
        for byte in iterbytes(self.requests):
            a.dataReceived(byte)
        value = b.value()
        self.assertEqual(value, b'')
        self.assertEqual(1, len(a.requests))
        while a.requests:
            self.assertEqual(1, len(a.requests))
            request = a.requests[0].original
            request.delayedProcess()
        value = b.value()
        self.assertResponseEquals(value, self.expectedResponses)

    def test_pipeliningReadLimit(self):
        """
        When pipelined requests are received, we will optimistically continue
        receiving data up to a specified limit, then pause the transport.

        @see: L{http.HTTPChannel._optimisticEagerReadSize}
        """
        b = StringTransport()
        a = http.HTTPChannel()
        a.requestFactory = DelayedHTTPHandlerProxy
        a.makeConnection(b)
        underLimit = a._optimisticEagerReadSize // len(self.requests)
        for x in range(1, underLimit + 1):
            a.dataReceived(self.requests)
            self.assertEqual(b.producerState, 'producing', 'state was {state!r} after {x} iterations'.format(state=b.producerState, x=x))
        a.dataReceived(self.requests)
        self.assertEquals(b.producerState, 'paused')