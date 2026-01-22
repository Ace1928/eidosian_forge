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
class LengthEnforcingConsumerTests(TestCase):
    """
    Tests for L{LengthEnforcingConsumer}.
    """

    def setUp(self):
        self.result = Deferred()
        self.producer = StringProducer(10)
        self.transport = StringTransport()
        self.enforcer = LengthEnforcingConsumer(self.producer, self.transport, self.result)

    def test_write(self):
        """
        L{LengthEnforcingConsumer.write} calls the wrapped consumer's C{write}
        method with the bytes it is passed as long as there are fewer of them
        than the C{length} attribute indicates remain to be received.
        """
        self.enforcer.write(b'abc')
        self.assertEqual(self.transport.value(), b'abc')
        self.transport.clear()
        self.enforcer.write(b'def')
        self.assertEqual(self.transport.value(), b'def')

    def test_finishedEarly(self):
        """
        L{LengthEnforcingConsumer._noMoreWritesExpected} raises
        L{WrongBodyLength} if it is called before the indicated number of bytes
        have been written.
        """
        self.enforcer.write(b'x' * 9)
        self.assertRaises(WrongBodyLength, self.enforcer._noMoreWritesExpected)

    def test_writeTooMany(self, _unregisterAfter=False):
        """
        If it is called with a total number of bytes exceeding the indicated
        limit passed to L{LengthEnforcingConsumer.__init__},
        L{LengthEnforcingConsumer.write} fires the L{Deferred} with a
        L{Failure} wrapping a L{WrongBodyLength} and also calls the
        C{stopProducing} method of the producer.
        """
        self.enforcer.write(b'x' * 10)
        self.assertFalse(self.producer.stopped)
        self.enforcer.write(b'x')
        self.assertTrue(self.producer.stopped)
        if _unregisterAfter:
            self.enforcer._noMoreWritesExpected()
        return self.assertFailure(self.result, WrongBodyLength)

    def test_writeAfterNoMoreExpected(self):
        """
        If L{LengthEnforcingConsumer.write} is called after
        L{LengthEnforcingConsumer._noMoreWritesExpected}, it calls the
        producer's C{stopProducing} method and raises L{ExcessWrite}.
        """
        self.enforcer.write(b'x' * 10)
        self.enforcer._noMoreWritesExpected()
        self.assertFalse(self.producer.stopped)
        self.assertRaises(ExcessWrite, self.enforcer.write, b'x')
        self.assertTrue(self.producer.stopped)

    def test_finishedLate(self):
        """
        L{LengthEnforcingConsumer._noMoreWritesExpected} does nothing (in
        particular, it does not raise any exception) if called after too many
        bytes have been passed to C{write}.
        """
        return self.test_writeTooMany(True)

    def test_finished(self):
        """
        If L{LengthEnforcingConsumer._noMoreWritesExpected} is called after
        the correct number of bytes have been written it returns L{None}.
        """
        self.enforcer.write(b'x' * 10)
        self.assertIdentical(self.enforcer._noMoreWritesExpected(), None)

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