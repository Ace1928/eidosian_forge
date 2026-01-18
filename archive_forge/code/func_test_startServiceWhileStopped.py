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
def test_startServiceWhileStopped(self):
    """
        When L{ClientService} is stopped - that is,
        L{ClientService.stopService} has been called and the L{Deferred} it
        returns has fired - calling L{startService} will cause a new connection
        to be made, and new calls to L{whenConnected} to succeed.
        """
    cq, service = self.makeReconnector(fireImmediately=False)
    stopped = service.stopService()
    self.successResultOf(stopped)
    self.failureResultOf(service.whenConnected(), CancelledError)
    service.startService()
    cq.connectQueue[-1].callback(None)
    self.assertIdentical(cq.applicationProtocols[-1], self.successResultOf(service.whenConnected()))