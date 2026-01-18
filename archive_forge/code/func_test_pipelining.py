from twisted.internet.defer import Deferred, DeferredList, TimeoutError, gatherResults
from twisted.internet.error import ConnectionDone
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransportWithDisconnection
from twisted.protocols.memcache import (
from twisted.trial.unittest import TestCase
def test_pipelining(self):
    """
        Multiple requests can be sent subsequently to the server, and the
        protocol orders the responses correctly and dispatch to the
        corresponding client command.
        """
    d1 = self.proto.get(b'foo')
    d1.addCallback(self.assertEqual, (0, b'bar'))
    d2 = self.proto.set(b'bar', b'spamspamspam')
    d2.addCallback(self.assertEqual, True)
    d3 = self.proto.get(b'egg')
    d3.addCallback(self.assertEqual, (0, b'spam'))
    self.assertEqual(self.transport.value(), b'get foo\r\nset bar 0 0 12\r\nspamspamspam\r\nget egg\r\n')
    self.proto.dataReceived(b'VALUE foo 0 3\r\nbar\r\nEND\r\nSTORED\r\nVALUE egg 0 4\r\nspam\r\nEND\r\n')
    return gatherResults([d1, d2, d3])