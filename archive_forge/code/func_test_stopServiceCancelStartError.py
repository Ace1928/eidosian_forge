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
def test_stopServiceCancelStartError(self):
    """
        L{StreamServerEndpointService.stopService} cancels the L{Deferred}
        returned by C{listen} if it has not fired yet.  An error will be logged
        if the resulting exception is not L{CancelledError}.
        """
    self.fakeServer.cancelException = ZeroDivisionError()
    self.svc.privilegedStartService()
    result = self.svc.stopService()
    l = []
    result.addCallback(l.append)
    self.assertEqual(l, [None])
    stoppingErrors = self.flushLoggedErrors(ZeroDivisionError)
    self.assertEqual(len(stoppingErrors), 1)