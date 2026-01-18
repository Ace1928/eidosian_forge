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
def test_sendRequestBodyWithError(self):
    """
        If the L{Deferred} returned from the C{startProducing} method of the
        L{IBodyProducer} passed to L{Request} fires with a L{Failure}, the
        L{Deferred} returned from L{Request.writeTo} fails with that
        L{Failure}.
        """
    producer = StringProducer(5)
    request = Request(b'POST', b'/bar', _boringHeaders, producer)
    writeDeferred = request.writeTo(self.transport)
    self.assertIdentical(self.transport.producer, producer)
    self.assertTrue(self.transport.streaming)
    producer.consumer.write(b'ab')
    self.assertEqual(self.transport.value(), b'POST /bar HTTP/1.1\r\nConnection: close\r\nContent-Length: 5\r\nHost: example.com\r\n\r\nab')
    self.assertFalse(self.transport.disconnecting)
    producer.finished.errback(Failure(ArbitraryException()))
    self.assertFalse(self.transport.disconnecting)
    self.assertIdentical(self.transport.producer, None)
    return self.assertFailure(writeDeferred, ArbitraryException)