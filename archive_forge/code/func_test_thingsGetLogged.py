import builtins
from io import StringIO
from zope.interface import Interface, implementedBy, implementer
from twisted.internet import address, defer, protocol, reactor, task
from twisted.internet.testing import StringTransport, StringTransportWithDisconnection
from twisted.protocols import policies
from twisted.trial import unittest
def test_thingsGetLogged(self):
    """
        Check the output produced by L{policies.TrafficLoggingFactory}.
        """
    wrappedFactory = Server()
    wrappedFactory.protocol = WriteSequenceEchoProtocol
    t = StringTransportWithDisconnection()
    f = TestLoggingFactory(wrappedFactory, 'test')
    p = f.buildProtocol(('1.2.3.4', 5678))
    t.protocol = p
    p.makeConnection(t)
    v = f.openFile.getvalue()
    self.assertIn('*', v)
    self.assertFalse(t.value())
    p.dataReceived(b'here are some bytes')
    v = f.openFile.getvalue()
    self.assertIn('C 1: {!r}'.format(b'here are some bytes'), v)
    self.assertIn('S 1: {!r}'.format(b'here are some bytes'), v)
    self.assertEqual(t.value(), b'here are some bytes')
    t.clear()
    p.dataReceived(b'prepare for vector! to the extreme')
    v = f.openFile.getvalue()
    self.assertIn('SV 1: {!r}'.format([b'prepare for vector! to the extreme']), v)
    self.assertEqual(t.value(), b'prepare for vector! to the extreme')
    p.loseConnection()
    v = f.openFile.getvalue()
    self.assertIn('ConnectionDone', v)