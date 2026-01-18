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
def test_postRequest(self):
    """
        Send a POST request and confirm that the data is safely transferred.
        """
    connection = H2Connection()
    connection.requestFactory = DummyHTTPHandlerProxy
    _, transport = self.connectAndReceive(connection, self.postRequestHeaders, self.postRequestData)

    def validate(streamID):
        frames = framesFromBytes(transport.value())
        self.assertEqual(len(frames), 4)
        self.assertTrue(all((f.stream_id == 1 for f in frames[-3:])))
        self.assertTrue(isinstance(frames[-3], hyperframe.frame.HeadersFrame))
        self.assertTrue(isinstance(frames[-2], hyperframe.frame.DataFrame))
        self.assertTrue(isinstance(frames[-1], hyperframe.frame.DataFrame))
        self.assertEqual(dict(frames[-3].data), dict(self.postResponseHeaders))
        self.assertEqual(frames[-2].data, self.postResponseData)
        self.assertEqual(frames[-1].data, b'')
        self.assertTrue('END_STREAM' in frames[-1].flags)
    return connection._streamCleanupCallbacks[1].addCallback(validate)