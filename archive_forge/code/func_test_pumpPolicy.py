from zope.interface import implementer
from twisted.internet import defer, interfaces, reactor
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IAddress, IPullProducer, IPushProducer
from twisted.internet.protocol import Protocol
from twisted.protocols import basic, loopback
from twisted.trial import unittest
def test_pumpPolicy(self):
    """
        The callable passed as the value for the C{pumpPolicy} parameter to
        L{loopbackAsync} is called with a L{_LoopbackQueue} of pending bytes
        and a protocol to which they should be delivered.
        """
    pumpCalls = []

    def dummyPolicy(queue, target):
        bytes = []
        while queue:
            bytes.append(queue.get())
        pumpCalls.append((target, bytes))
    client = Protocol()
    server = Protocol()
    finished = loopback.loopbackAsync(server, client, dummyPolicy)
    self.assertEqual(pumpCalls, [])
    client.transport.write(b'foo')
    client.transport.write(b'bar')
    server.transport.write(b'baz')
    server.transport.write(b'quux')
    server.transport.loseConnection()

    def cbComplete(ignored):
        self.assertEqual(pumpCalls, [(client, [b'baz', b'quux', None]), (server, [b'foo', b'bar'])])
    finished.addCallback(cbComplete)
    return finished