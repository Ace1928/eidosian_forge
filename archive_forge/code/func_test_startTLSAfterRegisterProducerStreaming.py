from twisted.internet import interfaces
from twisted.internet.test.connectionmixins import (
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.internet.test.test_tcp import TCPCreator
from twisted.internet.test.test_tls import (
from twisted.trial import unittest
from zope.interface import implementer
def test_startTLSAfterRegisterProducerStreaming(self):
    """
        When a streaming producer is registered, and then startTLS is called,
        the producer is re-registered with the C{TLSMemoryBIOProtocol}.
        """
    self.startTLSAfterRegisterProducer(True)