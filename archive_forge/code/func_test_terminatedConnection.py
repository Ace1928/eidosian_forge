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
def test_terminatedConnection(self):
    """
        When a GoAway frame is received, the L{H2Connection} and L{H2Stream}
        objects tear down all outstanding L{http.Request} objects and stop all
        writing.
        """
    connection = H2Connection()
    connection.requestFactory = DummyProducerHandlerProxy
    frameFactory, transport = self.connectAndReceive(connection, self.getRequestHeaders, [])
    request = connection.streams[1]._request.original
    request.write(b'first chunk')
    request.write(b'second chunk')
    cleanupCallback = connection._streamCleanupCallbacks[1]
    connection.dataReceived(frameFactory.buildGoAwayFrame(lastStreamID=0).serialize())
    self.assertTrue(request._disconnected)
    self.assertTrue(request.channel is None)
    self.assertFalse(connection._stillProducing)

    def validate(streamID):
        frames = framesFromBytes(transport.value())
        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[1].stream_id, 1)
        self.assertTrue(isinstance(frames[1], hyperframe.frame.HeadersFrame))
    return cleanupCallback.addCallback(validate)