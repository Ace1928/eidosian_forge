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
def test_timeoutResetByRequestData(self):
    """
        When a L{H2Connection} receives data, the timeout is reset.
        """
    frameFactory = FrameFactory()
    initialData = b''
    reactor, conn, transport = self.initiateH2Connection(initialData, requestFactory=DummyHTTPHandler)
    for byte in iterbytes(frameFactory.clientConnectionPreface()):
        conn.dataReceived(byte)
        reactor.advance(99)
        self.assertFalse(transport.disconnecting)
    reactor.advance(2)
    self.assertTimedOut(transport.value(), frameCount=2, errorCode=h2.errors.ErrorCodes.NO_ERROR, lastStreamID=0)
    self.assertTrue(transport.disconnecting)