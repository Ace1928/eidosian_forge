from typing import Optional
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.internet.defer import CancelledError, Deferred, fail, succeed
from twisted.internet.error import ConnectionDone, ConnectionLost
from twisted.internet.interfaces import IConsumer, IPushProducer
from twisted.internet.protocol import Protocol
from twisted.internet.testing import (
from twisted.logger import globalLogPublisher
from twisted.protocols.basic import LineReceiver
from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase
from twisted.web._newclient import (
from twisted.web.client import (
from twisted.web.http import _DataLoss
from twisted.web.http_headers import Headers
from twisted.web.iweb import IBodyProducer, IResponse
from twisted.web.test.requesthelper import (
def test_transportResumed(self):
    """
        L{Response.deliverBody} resumes the HTTP connection's transport
        after passing it to the consumer's C{makeConnection} method.
        """
    transportState = []

    class ListConsumer(Protocol):

        def makeConnection(self, transport):
            transportState.append(transport.producerState)
    transport = StringTransport()
    transport.pauseProducing()
    protocol = ListConsumer()
    response = justTransportResponse(transport)
    self.assertEqual(transport.producerState, 'paused')
    response.deliverBody(protocol)
    self.assertEqual(transportState, ['paused'])
    self.assertEqual(transport.producerState, 'producing')