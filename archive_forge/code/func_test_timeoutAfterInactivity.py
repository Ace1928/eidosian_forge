import itertools
from zope.interface import directlyProvides, providedBy
from twisted.internet import defer, error, reactor, task
from twisted.internet.address import IPv4Address
from twisted.internet.testing import MemoryReactorClock, StringTransport
from twisted.python import failure
from twisted.python.compat import iterbytes
from twisted.test.test_internet import DummyProducer
from twisted.trial import unittest
from twisted.web import http
from twisted.web.test.test_http import (
def test_timeoutAfterInactivity(self):
    """
        When a L{H2Connection} does not receive any data for more than the
        time out interval, it closes the connection cleanly.
        """
    frameFactory = FrameFactory()
    initialData = frameFactory.clientConnectionPreface()
    reactor, conn, transport = self.initiateH2Connection(initialData, requestFactory=DummyHTTPHandler)
    preamble = transport.value()
    reactor.advance(99)
    self.assertEqual(preamble, transport.value())
    self.assertFalse(transport.disconnecting)
    reactor.advance(2)
    self.assertTimedOut(transport.value(), frameCount=2, errorCode=h2.errors.ErrorCodes.NO_ERROR, lastStreamID=0)
    self.assertTrue(transport.disconnecting)