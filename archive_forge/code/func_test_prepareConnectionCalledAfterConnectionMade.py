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
def test_prepareConnectionCalledAfterConnectionMade(self):
    """
        The C{prepareConnection} callback is invoked only once a connection is
        made.
        """
    prepares = [0]

    def prepareConnection(_proto):
        prepares[0] += 1
    clock = Clock()
    cq, service = self.makeReconnector(prepareConnection=prepareConnection, fireImmediately=False, clock=clock)
    cq.connectQueue[0].errback(Exception('connection attempt failed'))
    self.assertEqual(0, prepares[0])
    clock.advance(AT_LEAST_ONE_ATTEMPT)
    cq.connectQueue[1].callback(None)
    self.assertEqual(1, prepares[0])