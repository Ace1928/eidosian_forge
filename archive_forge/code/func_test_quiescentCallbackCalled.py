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
def test_quiescentCallbackCalled(self):
    """
        If after a response is done the {HTTP11ClientProtocol} stays open and
        returns to QUIESCENT state, all per-request state is reset and the
        C{quiescentCallback} is called with the protocol instance.

        This is useful for implementing a persistent connection pool.

        The C{quiescentCallback} is called *before* the response-receiving
        protocol's C{connectionLost}, so that new requests triggered by end of
        first request can re-use a persistent connection.
        """
    quiescentResult = []

    def callback(p):
        self.assertEqual(p, protocol)
        self.assertEqual(p.state, 'QUIESCENT')
        quiescentResult.append(p)
    transport = StringTransport()
    protocol = HTTP11ClientProtocol(callback)
    protocol.makeConnection(transport)
    requestDeferred = protocol.request(Request(b'GET', b'/', _boringHeaders, None, persistent=True))
    protocol.dataReceived(b'HTTP/1.1 200 OK\r\nContent-length: 3\r\n\r\n')
    self.assertEqual(quiescentResult, [])
    result = []
    requestDeferred.addCallback(result.append)
    response = result[0]
    bodyProtocol = AccumulatingProtocol()
    bodyProtocol.closedDeferred = Deferred()
    bodyProtocol.closedDeferred.addCallback(lambda ign: quiescentResult.append('response done'))
    response.deliverBody(bodyProtocol)
    protocol.dataReceived(b'abc')
    bodyProtocol.closedReason.trap(ResponseDone)
    self.assertEqual(quiescentResult, [protocol, 'response done'])
    self.assertEqual(protocol._parser, None)
    self.assertEqual(protocol._finishedRequest, None)
    self.assertEqual(protocol._currentRequest, None)
    self.assertEqual(protocol._transportProxy, None)
    self.assertEqual(protocol._responseDeferred, None)