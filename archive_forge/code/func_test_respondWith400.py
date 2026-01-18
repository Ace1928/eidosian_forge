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
def test_respondWith400(self):
    """
        Triggering the call to L{H2Stream._respondToBadRequestAndDisconnect}
        leads to a 400 error being sent automatically and the stream being torn
        down.
        """
    connection = H2Connection()
    connection.requestFactory = DummyProducerHandlerProxy
    _, transport = self.connectAndReceive(connection, self.getRequestHeaders, [])
    stream = connection.streams[1]
    request = stream._request.original
    cleanupCallback = connection._streamCleanupCallbacks[1]
    stream._respondToBadRequestAndDisconnect()
    self.assertTrue(request._disconnected)
    self.assertTrue(request.channel is None)

    def validate(streamID):
        frames = framesFromBytes(transport.value())
        self.assertEqual(len(frames), 2)
        self.assertTrue(isinstance(frames[1], hyperframe.frame.HeadersFrame))
        self.assertEqual(frames[1].data, [(b':status', b'400')])
        self.assertTrue('END_STREAM' in frames[-1].flags)
    return cleanupCallback.addCallback(validate)