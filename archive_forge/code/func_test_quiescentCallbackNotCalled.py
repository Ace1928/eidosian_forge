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
def test_quiescentCallbackNotCalled(self):
    """
        If after a response is done the {HTTP11ClientProtocol} returns a
        C{Connection: close} header in the response, the C{quiescentCallback}
        is not called and the connection is lost.
        """
    quiescentResult = []
    transport = StringTransport()
    protocol = HTTP11ClientProtocol(quiescentResult.append)
    protocol.makeConnection(transport)
    requestDeferred = protocol.request(Request(b'GET', b'/', _boringHeaders, None, persistent=True))
    protocol.dataReceived(b'HTTP/1.1 200 OK\r\nContent-length: 0\r\nConnection: close\r\n\r\n')
    result = []
    requestDeferred.addCallback(result.append)
    response = result[0]
    bodyProtocol = AccumulatingProtocol()
    response.deliverBody(bodyProtocol)
    bodyProtocol.closedReason.trap(ResponseDone)
    self.assertEqual(quiescentResult, [])
    self.assertTrue(transport.disconnecting)