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
def test_responseBodyFinishedWhenConnectionLostWhenContentLengthIsUnknown(self):
    """
        If the length of the response body is unknown, the protocol passed to
        the response's C{deliverBody} method has its C{connectionLost}
        method called with a L{Failure} wrapping a L{PotentialDataLoss}
        exception.
        """
    requestDeferred = self.protocol.request(Request(b'GET', b'/', _boringHeaders, None))
    self.protocol.dataReceived(b'HTTP/1.1 200 OK\r\n\r\n')
    result = []
    requestDeferred.addCallback(result.append)
    response = result[0]
    protocol = AccumulatingProtocol()
    response.deliverBody(protocol)
    self.protocol.dataReceived(b'foo')
    self.protocol.dataReceived(b'bar')
    self.assertEqual(protocol.data, b'foobar')
    self.protocol.connectionLost(Failure(ConnectionDone('low-level transport disconnected')))
    protocol.closedReason.trap(PotentialDataLoss)