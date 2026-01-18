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
def test_stopServiceImmediately(self):
    """
        When L{TimerService.stopService} is called while a call isn't in progress.
        the L{Deferred} returned has already been fired.
        """
    self.timer.startService()
    self.deferred.callback(object())
    d = self.timer.stopService()
    self.assertIdentical(self.successResultOf(d), None)