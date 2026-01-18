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
def test_sendAccordingToPriority(self):
    """
        Data in responses is interleaved according to HTTP/2 priorities.
        """
    f = FrameFactory()
    b = StringTransport()
    a = H2Connection()
    a.requestFactory = ChunkedHTTPHandlerProxy
    getRequestHeaders = self.getRequestHeaders
    getRequestHeaders[2] = (':path', '/chunked/4')
    frames = [buildRequestFrames(getRequestHeaders, [], f, streamID) for streamID in [1, 3, 5]]
    frames[0][0].flags.add('PRIORITY')
    frames[0][0].stream_weight = 64
    frames[1][0].flags.add('PRIORITY')
    frames[1][0].stream_weight = 32
    priorityFrame = f.buildPriorityFrame(streamID=5, weight=16, dependsOn=1, exclusive=True)
    frames[2].insert(0, priorityFrame)
    frames = itertools.chain.from_iterable(frames)
    requestBytes = f.clientConnectionPreface()
    requestBytes += b''.join((frame.serialize() for frame in frames))
    a.makeConnection(b)
    for byte in iterbytes(requestBytes):
        a.dataReceived(byte)

    def validate(results):
        frames = framesFromBytes(b.value())
        self.assertEqual(len(frames), 19)
        streamIDs = [f.stream_id for f in frames if isinstance(f, hyperframe.frame.DataFrame)]
        expectedOrder = [1, 3, 1, 1, 3, 1, 1, 3, 5, 3, 5, 3, 5, 5, 5]
        self.assertEqual(streamIDs, expectedOrder)
    return defer.DeferredList(list(a._streamCleanupCallbacks.values())).addCallback(validate)