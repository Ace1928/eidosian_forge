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
def test_writeSequenceForChannels(self):
    """
        L{H2Stream} objects can send a series of frames via C{writeSequence}.
        """
    connection = H2Connection()
    connection.requestFactory = DelayedHTTPHandlerProxy
    _, transport = self.connectAndReceive(connection, self.getRequestHeaders, [])
    stream = connection.streams[1]
    request = stream._request.original
    request.setResponseCode(200)
    stream.writeSequence([b'Hello', b',', b'world!'])
    request.finish()
    completionDeferred = connection._streamCleanupCallbacks[1]

    def validate(streamID):
        frames = framesFromBytes(transport.value())
        self.assertTrue('END_STREAM' in frames[-1].flags)
        dataChunks = [f.data for f in frames if isinstance(f, hyperframe.frame.DataFrame)]
        self.assertEqual(dataChunks, [b'Hello', b',', b'world!', b''])
    return completionDeferred.addCallback(validate)