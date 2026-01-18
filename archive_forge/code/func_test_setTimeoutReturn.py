import builtins
from io import StringIO
from zope.interface import Interface, implementedBy, implementer
from twisted.internet import address, defer, protocol, reactor, task
from twisted.internet.testing import StringTransport, StringTransportWithDisconnection
from twisted.protocols import policies
from twisted.trial import unittest
def test_setTimeoutReturn(self):
    """
        setTimeout should return the value of the previous timeout.
        """
    self.proto.timeOut = 5
    self.assertEqual(self.proto.setTimeout(10), 5)
    self.assertEqual(self.proto.setTimeout(None), 10)
    self.assertIsNone(self.proto.setTimeout(1))
    self.assertEqual(self.proto.timeOut, 1)
    self.proto.setTimeout(None)