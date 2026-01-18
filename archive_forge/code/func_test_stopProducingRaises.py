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
def test_stopProducingRaises(self):
    """
        If L{LengthEnforcingConsumer.write} calls the producer's
        C{stopProducing} because too many bytes were written and the
        C{stopProducing} method raises an exception, the exception is logged
        and the L{LengthEnforcingConsumer} still errbacks the finished
        L{Deferred}.
        """

    def brokenStopProducing():
        StringProducer.stopProducing(self.producer)
        raise ArbitraryException('stopProducing is busted')
    self.producer.stopProducing = brokenStopProducing

    def cbFinished(ignored):
        self.assertEqual(len(self.flushLoggedErrors(ArbitraryException)), 1)
    d = self.test_writeTooMany()
    d.addCallback(cbFinished)
    return d