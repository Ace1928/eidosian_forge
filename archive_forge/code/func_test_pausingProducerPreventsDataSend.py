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
def test_pausingProducerPreventsDataSend(self):
    """
        L{H2Connection} can be paused by its consumer. When paused it stops
        sending data to the transport.
        """
    f = FrameFactory()
    b = StringTransport()
    a = H2Connection()
    a.requestFactory = DummyHTTPHandlerProxy
    frames = buildRequestFrames(self.getRequestHeaders, [], f)
    requestBytes = f.clientConnectionPreface()
    requestBytes += b''.join((f.serialize() for f in frames))
    a.makeConnection(b)
    b.registerProducer(a, True)
    for byte in iterbytes(requestBytes):
        a.dataReceived(byte)
    a.pauseProducing()
    cleanupCallback = a._streamCleanupCallbacks[1]

    def validateNotSent(*args):
        frames = framesFromBytes(b.value())
        self.assertEqual(len(frames), 2)
        self.assertFalse(isinstance(frames[-1], hyperframe.frame.DataFrame))
        a.resumeProducing()
        a.resumeProducing()
        a.resumeProducing()
        a.resumeProducing()
        a.resumeProducing()
        return cleanupCallback

    def validateComplete(*args):
        frames = framesFromBytes(b.value())
        self.assertEqual(len(frames), 4)
        self.assertTrue('END_STREAM' in frames[-1].flags)
    d = task.deferLater(reactor, 0.01, validateNotSent)
    d.addCallback(validateComplete)
    return d