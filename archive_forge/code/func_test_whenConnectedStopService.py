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
def test_whenConnectedStopService(self):
    """
        L{ClientService.whenConnected} returns a L{Deferred} that fails when
        L{ClientService.stopService} is called.
        """
    clock = Clock()
    cq, service = self.makeReconnector(fireImmediately=False, clock=clock)
    a = service.whenConnected()
    b = service.whenConnected()
    c = service.whenConnected(failAfterFailures=1)
    self.assertNoResult(a)
    self.assertNoResult(b)
    self.assertNoResult(c)
    service.stopService()
    clock.advance(AT_LEAST_ONE_ATTEMPT)
    self.failureResultOf(a, CancelledError)
    self.failureResultOf(b, CancelledError)
    self.failureResultOf(c, CancelledError)