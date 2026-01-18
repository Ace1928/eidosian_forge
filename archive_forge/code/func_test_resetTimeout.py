import builtins
from io import StringIO
from zope.interface import Interface, implementedBy, implementer
from twisted.internet import address, defer, protocol, reactor, task
from twisted.internet.testing import StringTransport, StringTransportWithDisconnection
from twisted.protocols import policies
from twisted.trial import unittest
def test_resetTimeout(self):
    """
        Check that setting a new value for timeout cancel the previous value
        and install a new timeout.
        """
    self.proto.timeOut = None
    self.proto.makeConnection(StringTransport())
    self.proto.setTimeout(1)
    self.assertEqual(self.proto.timeOut, 1)
    self.clock.pump([0, 0.9])
    self.assertFalse(self.proto.timedOut)
    self.clock.pump([0, 0.2])
    self.assertTrue(self.proto.timedOut)