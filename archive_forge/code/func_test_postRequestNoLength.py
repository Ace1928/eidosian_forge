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
def test_postRequestNoLength(self):
    """
        Send a POST request without length and confirm that the data is safely
        transferred.
        """
    postResponseHeaders = [(b':status', b'200'), (b'request', b'/post_endpoint'), (b'command', b'POST'), (b'version', b'HTTP/2'), (b'content-length', b'38')]
    postResponseData = b"'''\nNone\nhello world, it's http/2!'''\n"
    postRequestHeaders = [(x, y) for x, y in self.postRequestHeaders if x != b'content-length']
    connection = H2Connection()
    connection.requestFactory = DummyHTTPHandlerProxy
    _, transport = self.connectAndReceive(connection, postRequestHeaders, self.postRequestData)

    def validate(streamID):
        frames = framesFromBytes(transport.value())
        self.assertEqual(len(frames), 4)
        self.assertTrue(all((f.stream_id == 1 for f in frames[-3:])))
        self.assertTrue(isinstance(frames[-3], hyperframe.frame.HeadersFrame))
        self.assertTrue(isinstance(frames[-2], hyperframe.frame.DataFrame))
        self.assertTrue(isinstance(frames[-1], hyperframe.frame.DataFrame))
        self.assertEqual(dict(frames[-3].data), dict(postResponseHeaders))
        self.assertEqual(frames[-2].data, postResponseData)
        self.assertEqual(frames[-1].data, b'')
        self.assertTrue('END_STREAM' in frames[-1].flags)
    return connection._streamCleanupCallbacks[1].addCallback(validate)