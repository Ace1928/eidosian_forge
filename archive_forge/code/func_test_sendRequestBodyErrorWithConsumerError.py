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
def test_sendRequestBodyErrorWithConsumerError(self):
    """
        Though there should be no way for the internal C{finishedConsuming}
        L{Deferred} in L{Request._writeToBodyProducerContentLength} to fire a
        L{Failure} after the C{finishedProducing} L{Deferred} has fired, in
        case this does happen, the error should be logged with a message about
        how there's probably a bug in L{Request}.

        This is a whitebox test.
        """
    logObserver = EventLoggingObserver.createWithCleanup(self, globalLogPublisher)
    producer = StringProducer(3)
    request = Request(b'POST', b'/bar', _boringHeaders, producer)
    request.writeTo(self.transport)
    finishedConsuming = producer.consumer._finished
    producer.consumer.write(b'abc')
    producer.finished.callback(None)
    finishedConsuming.errback(ArbitraryException())
    event = logObserver[0]
    self.assertIn('log_failure', event)
    f = event['log_failure']
    self.assertIsInstance(f.value, ArbitraryException)
    self.assertEqual(len(self.flushLoggedErrors(ArbitraryException)), 1)