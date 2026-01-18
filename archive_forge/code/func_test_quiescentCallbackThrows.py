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
def test_quiescentCallbackThrows(self):
    """
        If C{quiescentCallback} throws an exception, the error is logged and
        protocol is disconnected.
        """

    def callback(p):
        raise ZeroDivisionError()
    logObserver = EventLoggingObserver.createWithCleanup(self, globalLogPublisher)
    transport = StringTransport()
    protocol = HTTP11ClientProtocol(callback)
    protocol.makeConnection(transport)
    requestDeferred = protocol.request(Request(b'GET', b'/', _boringHeaders, None, persistent=True))
    protocol.dataReceived(b'HTTP/1.1 200 OK\r\nContent-length: 0\r\n\r\n')
    result = []
    requestDeferred.addCallback(result.append)
    response = result[0]
    bodyProtocol = AccumulatingProtocol()
    response.deliverBody(bodyProtocol)
    bodyProtocol.closedReason.trap(ResponseDone)
    self.assertEquals(1, len(logObserver))
    event = logObserver[0]
    f = event['log_failure']
    self.assertIsInstance(f.value, ZeroDivisionError)
    self.flushLoggedErrors(ZeroDivisionError)
    self.assertTrue(transport.disconnecting)