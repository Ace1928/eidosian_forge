from twisted.internet import interfaces
from twisted.internet.test.connectionmixins import (
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.internet.test.test_tcp import TCPCreator
from twisted.internet.test.test_tls import (
from twisted.trial import unittest
from zope.interface import implementer
class ProducerTestsMixin(ReactorBuilder, TLSMixin, ContextGeneratingMixin):
    """
    Test the new TLS code integrates C{TLSMemoryBIOProtocol} correctly.
    """
    if not _newtls:
        skip = 'Could not import twisted.internet._newtls'

    def test_producerSSLFromStart(self):
        """
        C{registerProducer} and C{unregisterProducer} on TLS transports
        created as SSL from the get go are passed to the
        C{TLSMemoryBIOProtocol}, not the underlying transport directly.
        """
        result = []
        producer = FakeProducer()
        runProtocolsWithReactor(self, ConnectableProtocol(), ProducerProtocol(producer, result), SSLCreator())
        self.assertEqual(result, [producer, None])

    def test_producerAfterStartTLS(self):
        """
        C{registerProducer} and C{unregisterProducer} on TLS transports
        created by C{startTLS} are passed to the C{TLSMemoryBIOProtocol}, not
        the underlying transport directly.
        """
        result = []
        producer = FakeProducer()
        runProtocolsWithReactor(self, ConnectableProtocol(), ProducerProtocol(producer, result), StartTLSClientCreator())
        self.assertEqual(result, [producer, None])

    def startTLSAfterRegisterProducer(self, streaming):
        """
        When a producer is registered, and then startTLS is called,
        the producer is re-registered with the C{TLSMemoryBIOProtocol}.
        """
        clientContext = self.getClientContext()
        serverContext = self.getServerContext()
        result = []
        producer = FakeProducer()

        class RegisterTLSProtocol(ConnectableProtocol):

            def connectionMade(self):
                self.transport.registerProducer(producer, streaming)
                self.transport.startTLS(serverContext)
                if streaming:
                    result.append(self.transport.protocol._producer._producer)
                    result.append(self.transport.producer._producer)
                else:
                    result.append(self.transport.protocol._producer._producer._producer)
                    result.append(self.transport.producer._producer._producer)
                self.transport.unregisterProducer()
                self.transport.loseConnection()

        class StartTLSProtocol(ConnectableProtocol):

            def connectionMade(self):
                self.transport.startTLS(clientContext)
        runProtocolsWithReactor(self, RegisterTLSProtocol(), StartTLSProtocol(), TCPCreator())
        self.assertEqual(result, [producer, producer])

    def test_startTLSAfterRegisterProducerStreaming(self):
        """
        When a streaming producer is registered, and then startTLS is called,
        the producer is re-registered with the C{TLSMemoryBIOProtocol}.
        """
        self.startTLSAfterRegisterProducer(True)

    def test_startTLSAfterRegisterProducerNonStreaming(self):
        """
        When a non-streaming producer is registered, and then startTLS is
        called, the producer is re-registered with the
        C{TLSMemoryBIOProtocol}.
        """
        self.startTLSAfterRegisterProducer(False)