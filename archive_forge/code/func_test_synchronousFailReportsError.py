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
def test_synchronousFailReportsError(self):
    """
        Without the C{_raiseSynchronously} compatibility flag, failing
        immediately has the same behavior as failing later; it logs the error.
        """
    self.fakeServer.failImmediately = ZeroDivisionError()
    self.svc.startService()
    logged = self.flushLoggedErrors(ZeroDivisionError)
    self.assertEqual(len(logged), 1)