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
def test_streamProducingData(self):
    """
        The H2Stream data implements IPushProducer, and can have its data
        production controlled by the Request if the Request chooses to.
        """
    connection = H2Connection()
    connection.requestFactory = ConsumerDummyHandlerProxy
    _, transport = self.connectAndReceive(connection, self.postRequestHeaders, self.postRequestData)
    request = connection.streams[1]._request.original
    self.assertFalse(request._requestReceived)
    frames = framesFromBytes(transport.value())
    self.assertEqual(len(frames), 1)
    request.acceptData()
    self.assertTrue(request._requestReceived)
    self.assertTrue(request._data, b"hello world, it's http/2!")
    frames = framesFromBytes(transport.value())
    self.assertEqual(len(frames), 2)

    def validate(streamID):
        frames = framesFromBytes(transport.value())
        self.assertEqual(len(frames), 4)
        self.assertTrue('END_STREAM' in frames[-1].flags)
    return connection._streamCleanupCallbacks[1].addCallback(validate)