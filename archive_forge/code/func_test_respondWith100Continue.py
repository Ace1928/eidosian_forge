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
def test_respondWith100Continue(self):
    """
        Requests containing Expect: 100-continue cause provisional 100
        responses to be emitted.
        """
    connection = H2Connection()
    connection.requestFactory = DummyHTTPHandlerProxy
    headers = self.getRequestHeaders + [(b'expect', b'100-continue')]
    _, transport = self.connectAndReceive(connection, headers, [])

    def validate(streamID):
        frames = framesFromBytes(transport.value())
        self.assertEqual(len(frames), 5)
        self.assertTrue(all((f.stream_id == 1 for f in frames[1:])))
        self.assertTrue(isinstance(frames[1], hyperframe.frame.HeadersFrame))
        self.assertEqual(frames[1].data, [(b':status', b'100')])
        self.assertTrue('END_STREAM' in frames[-1].flags)
    return connection._streamCleanupCallbacks[1].addCallback(validate)