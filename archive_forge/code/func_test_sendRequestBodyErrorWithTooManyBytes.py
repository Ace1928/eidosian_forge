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
def test_sendRequestBodyErrorWithTooManyBytes(self):
    """
        If L{Request} is created with a C{bodyProducer} with a known length and
        the producer tries to produce more than than many bytes, the
        L{Deferred} returned by L{Request.writeTo} fires with a L{Failure}
        wrapping a L{WrongBodyLength} exception.
        """
    logObserver = EventLoggingObserver.createWithCleanup(self, globalLogPublisher)

    def finisher(producer):
        producer.finished.errback(ArbitraryException())
        event = logObserver[0]
        self.assertIn('log_failure', event)
        f = event['log_failure']
        self.assertIsInstance(f.value, ArbitraryException)
        errors = self.flushLoggedErrors(ArbitraryException)
        self.assertEqual(len(errors), 1)
    return self._sendRequestBodyWithTooManyBytesTest(finisher)