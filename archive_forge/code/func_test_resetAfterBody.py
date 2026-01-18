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
def test_resetAfterBody(self):
    """
        A client that immediately resets after sending the body causes Twisted
        to send no response.
        """
    frameFactory = FrameFactory()
    transport = StringTransport()
    a = H2Connection()
    a.requestFactory = DummyHTTPHandlerProxy
    requestBytes = frameFactory.clientConnectionPreface()
    requestBytes += buildRequestBytes(headers=self.getRequestHeaders, data=[], frameFactory=frameFactory)
    requestBytes += frameFactory.buildRstStreamFrame(streamID=1).serialize()
    a.makeConnection(transport)
    a.dataReceived(requestBytes)
    frames = framesFromBytes(transport.value())
    self.assertEqual(len(frames), 1)
    self.assertNotIn(1, a._streamCleanupCallbacks)