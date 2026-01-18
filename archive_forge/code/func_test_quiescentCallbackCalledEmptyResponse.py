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
def test_quiescentCallbackCalledEmptyResponse(self):
    """
        The quiescentCallback is called before the request C{Deferred} fires,
        in cases where the response has no body.
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
    requestDeferred.addCallback(quiescentResult.append)
    protocol.dataReceived(b'HTTP/1.1 200 OK\r\nContent-length: 0\r\n\r\n')
    self.assertEqual(len(quiescentResult), 2)
    self.assertIdentical(quiescentResult[0], protocol)
    self.assertIsInstance(quiescentResult[1], Response)