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
def test_startServiceUsesGlobalReactor(self):
    """
        L{TimerService.startService} uses L{internet._maybeGlobalReactor} to
        choose the reactor to pass to L{task.LoopingCall}
        uses the global reactor.
        """
    otherClock = task.Clock()

    def getOtherClock(maybeReactor):
        return otherClock
    self.patch(internet, '_maybeGlobalReactor', getOtherClock)
    self.timer.startService()
    self.assertIdentical(otherClock, self.timer._loop.clock)