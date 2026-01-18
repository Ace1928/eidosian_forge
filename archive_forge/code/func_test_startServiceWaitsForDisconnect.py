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
def test_startServiceWaitsForDisconnect(self):
    """
        When L{ClientService} is restarted after having been connected, it
        waits to start connecting until after having disconnected.
        """
    cq, service = self.makeReconnector()
    d = service.stopService()
    self.assertNoResult(d)
    protocol = cq.constructedProtocols[0]
    self.assertEqual(protocol.transport.disconnecting, True)
    service.startService()
    self.assertNoResult(d)
    self.assertEqual(len(cq.constructedProtocols), 1)
    protocol.connectionLost(Failure(Exception()))
    self.assertEqual(len(cq.constructedProtocols), 2)