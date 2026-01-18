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
def test_prepareConnectionThrows(self):
    """
        The connection attempt counts as a failure when the
        C{prepareConnection} callable throws.
        """
    clock = Clock()

    def prepareConnection(_proto):
        raise IndentationError()
    cq, service = self.makeReconnector(prepareConnection=prepareConnection, clock=clock)
    whenConnectedDeferred = service.whenConnected(failAfterFailures=2)
    self.assertNoResult(whenConnectedDeferred)
    clock.advance(AT_LEAST_ONE_ATTEMPT)
    self.assertNoResult(whenConnectedDeferred)
    clock.advance(AT_LEAST_ONE_ATTEMPT)
    self.assertIdentical(IndentationError, self.failureResultOf(whenConnectedDeferred).type)