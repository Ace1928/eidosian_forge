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
def test_requestBodyTimeoutFromFactory(self):
    """
        L{HTTPChannel} timeouts whenever data from a request body is not
        delivered to it in time, even when it gets built from a L{HTTPFactory}.
        """
    clock = Clock()
    factory = http.HTTPFactory(timeout=100, reactor=clock)
    factory.startFactory()
    protocol = factory.buildProtocol(None)
    transport = StringTransport()
    protocol = parametrizeTimeoutMixin(protocol, clock)
    self.assertEqual(protocol.timeOut, 100)
    protocol.makeConnection(transport)
    protocol.dataReceived(b'POST / HTTP/1.0\r\nContent-Length: 2\r\n\r\n')
    clock.advance(99)
    self.assertFalse(transport.disconnecting)
    clock.advance(2)
    self.assertTrue(transport.disconnecting)