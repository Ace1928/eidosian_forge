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
def test_prepareConnectionReturningADeferred(self):
    """
        The C{prepareConnection} callable returns a deferred and calls to
        L{ClientService.whenConnected} wait until it fires.
        """
    newProtocols = []
    newProtocolDeferred = Deferred()

    def prepareConnection(proto):
        newProtocols.append(proto)
        return newProtocolDeferred
    cq, service = self.makeReconnector(prepareConnection=prepareConnection)
    whenConnectedDeferred = service.whenConnected()
    self.assertNoResult(whenConnectedDeferred)
    newProtocolDeferred.callback(None)
    self.assertIdentical(cq.applicationProtocols[0], self.successResultOf(whenConnectedDeferred))