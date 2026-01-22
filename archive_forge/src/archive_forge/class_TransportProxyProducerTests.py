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
class TransportProxyProducerTests(TestCase):
    """
    Tests for L{TransportProxyProducer} which proxies the L{IPushProducer}
    interface of a transport.
    """

    def test_interface(self):
        """
        L{TransportProxyProducer} instances provide L{IPushProducer}.
        """
        self.assertTrue(verifyObject(IPushProducer, TransportProxyProducer(None)))

    def test_stopProxyingUnreferencesProducer(self):
        """
        L{TransportProxyProducer.stopProxying} drops the reference to the
        wrapped L{IPushProducer} provider.
        """
        transport = StringTransport()
        proxy = TransportProxyProducer(transport)
        self.assertIdentical(proxy._producer, transport)
        proxy.stopProxying()
        self.assertIdentical(proxy._producer, None)

    def test_resumeProducing(self):
        """
        L{TransportProxyProducer.resumeProducing} calls the wrapped
        transport's C{resumeProducing} method unless told to stop proxying.
        """
        transport = StringTransport()
        transport.pauseProducing()
        proxy = TransportProxyProducer(transport)
        self.assertEqual(transport.producerState, 'paused')
        proxy.resumeProducing()
        self.assertEqual(transport.producerState, 'producing')
        transport.pauseProducing()
        proxy.stopProxying()
        proxy.resumeProducing()
        self.assertEqual(transport.producerState, 'paused')

    def test_pauseProducing(self):
        """
        L{TransportProxyProducer.pauseProducing} calls the wrapped transport's
        C{pauseProducing} method unless told to stop proxying.
        """
        transport = StringTransport()
        proxy = TransportProxyProducer(transport)
        self.assertEqual(transport.producerState, 'producing')
        proxy.pauseProducing()
        self.assertEqual(transport.producerState, 'paused')
        transport.resumeProducing()
        proxy.stopProxying()
        proxy.pauseProducing()
        self.assertEqual(transport.producerState, 'producing')

    def test_stopProducing(self):
        """
        L{TransportProxyProducer.stopProducing} calls the wrapped transport's
        C{stopProducing} method unless told to stop proxying.
        """
        transport = StringTransport()
        proxy = TransportProxyProducer(transport)
        self.assertEqual(transport.producerState, 'producing')
        proxy.stopProducing()
        self.assertEqual(transport.producerState, 'stopped')
        transport = StringTransport()
        proxy = TransportProxyProducer(transport)
        proxy.stopProxying()
        proxy.stopProducing()
        self.assertEqual(transport.producerState, 'producing')

    def test_loseConnectionWhileProxying(self):
        """
        L{TransportProxyProducer.loseConnection} calls the wrapped transport's
        C{loseConnection}.
        """
        transport = StringTransportWithDisconnection()
        protocol = AccumulatingProtocol()
        protocol.makeConnection(transport)
        transport.protocol = protocol
        proxy = TransportProxyProducer(transport)
        self.assertTrue(transport.connected)
        self.assertEqual(transport.producerState, 'producing')
        proxy.loseConnection()
        self.assertEqual(transport.producerState, 'producing')
        self.assertFalse(transport.connected)

    def test_loseConnectionNotProxying(self):
        """
        L{TransportProxyProducer.loseConnection} does nothing when the
        proxy is not active.
        """
        transport = StringTransportWithDisconnection()
        protocol = AccumulatingProtocol()
        protocol.makeConnection(transport)
        transport.protocol = protocol
        proxy = TransportProxyProducer(transport)
        proxy.stopProxying()
        self.assertTrue(transport.connected)
        proxy.loseConnection()
        self.assertTrue(transport.connected)