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
def test_receiveResponseBeforeRequestGenerationDone(self):
    """
        If response bytes are delivered to L{HTTP11ClientProtocol} before the
        L{Deferred} returned by L{Request.writeTo} fires, those response bytes
        are parsed as part of the response.

        The connection is also closed, because we're in a confusing state, and
        therefore the C{quiescentCallback} isn't called.
        """
    quiescentResult = []
    transport = StringTransport()
    protocol = HTTP11ClientProtocol(quiescentResult.append)
    protocol.makeConnection(transport)
    request = SlowRequest()
    d = protocol.request(request)
    protocol.dataReceived(b'HTTP/1.1 200 OK\r\nX-Foo: bar\r\nContent-Length: 6\r\n\r\nfoobar')

    def cbResponse(response):
        p = AccumulatingProtocol()
        whenFinished = p.closedDeferred = Deferred()
        response.deliverBody(p)
        self.assertEqual(protocol.state, 'TRANSMITTING_AFTER_RECEIVING_RESPONSE')
        self.assertTrue(transport.disconnecting)
        self.assertEqual(quiescentResult, [])
        return whenFinished.addCallback(lambda ign: (response, p.data))
    d.addCallback(cbResponse)

    def cbAllResponse(result):
        response, body = result
        self.assertEqual(response.version, (b'HTTP', 1, 1))
        self.assertEqual(response.code, 200)
        self.assertEqual(response.phrase, b'OK')
        self.assertEqual(response.headers, Headers({b'x-foo': [b'bar']}))
        self.assertEqual(body, b'foobar')
        request.finished.callback(None)
    d.addCallback(cbAllResponse)
    return d