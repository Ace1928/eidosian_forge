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
def test_startServiceWhileStopping(self):
    """
        When L{ClientService} is stopping - that is,
        L{ClientService.stopService} has been called, but the L{Deferred} it
        returns has not fired yet - calling L{startService} will cause a new
        connection to be made, and new calls to L{whenConnected} to succeed.
        """
    cq, service = self.makeReconnector(fireImmediately=False)
    cq.connectQueue[0].callback(None)
    first = cq.constructedProtocols[0]
    stopped = service.stopService()
    self.assertNoResult(stopped)
    nextProtocol = service.whenConnected()
    self.assertNoResult(nextProtocol)
    service.startService()
    self.assertNoResult(nextProtocol)
    self.assertNoResult(stopped)
    self.assertEqual(first.transport.disconnecting, True)
    first.connectionLost(Failure(Exception()))
    self.successResultOf(stopped)
    cq.connectQueue[1].callback(None)
    self.assertEqual(len(cq.constructedProtocols), 2)
    self.assertIdentical(self.successResultOf(nextProtocol), cq.applicationProtocols[1])
    secondStopped = service.stopService()
    self.assertNoResult(secondStopped)