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
def test_sendRequestBodyWithTooFewBytes(self):
    """
        If L{Request} is created with a C{bodyProducer} with a known length and
        the producer does not produce that many bytes, the L{Deferred} returned
        by L{Request.writeTo} fires with a L{Failure} wrapping a
        L{WrongBodyLength} exception.
        """
    producer = StringProducer(3)
    request = Request(b'POST', b'/bar', _boringHeaders, producer)
    writeDeferred = request.writeTo(self.transport)
    producer.consumer.write(b'ab')
    producer.finished.callback(None)
    self.assertIdentical(self.transport.producer, None)
    return self.assertFailure(writeDeferred, WrongBodyLength)