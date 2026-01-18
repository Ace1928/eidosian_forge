import pickle
from zope.interface import implementer
from zope.interface.verify import verifyClass
from twisted.application import internet
from twisted.application.internet import (
from twisted.internet import task
from twisted.internet.defer import CancelledError, Deferred
from twisted.internet.interfaces import (
from twisted.internet.protocol import Factory, Protocol
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransport
from twisted.logger import formatEvent, globalLogPublisher
from twisted.python.failure import Failure
from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_stopServiceWhileConnected(self):
    """
        When the service is stopped, no further connect attempts are made.  The
        returned L{Deferred} fires when all outstanding connections have been
        stopped.
        """
    cq, service = self.makeReconnector()
    d = service.stopService()
    self.assertNoResult(d)
    protocol = cq.constructedProtocols[0]
    self.assertEqual(protocol.transport.disconnecting, True)
    protocol.connectionLost(Failure(Exception()))
    self.successResultOf(d)