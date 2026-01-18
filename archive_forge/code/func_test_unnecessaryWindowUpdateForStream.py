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
def test_unnecessaryWindowUpdateForStream(self):
    """
        When a WindowUpdate frame is received for a stream but no data is
        currently waiting, that stream is not marked as unblocked and the
        priority tree continues to assert that no stream can progress.
        """
    f = FrameFactory()
    transport = StringTransport()
    conn = H2Connection()
    conn.requestFactory = DummyHTTPHandlerProxy
    frames = []
    frames.append(f.buildHeadersFrame(headers=self.postRequestHeaders, streamID=1))
    frames.append(f.buildWindowUpdateFrame(streamID=1, increment=5))
    data = f.clientConnectionPreface()
    data += b''.join((f.serialize() for f in frames))
    conn.makeConnection(transport)
    conn.dataReceived(data)
    self.assertAllStreamsBlocked(conn)