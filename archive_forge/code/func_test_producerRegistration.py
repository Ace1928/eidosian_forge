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
def test_producerRegistration(self):
    """
        L{ChunkedEncoder.registerProducer} registers the given streaming
        producer with its transport and L{ChunkedEncoder.unregisterProducer}
        writes a zero-length chunk to its transport and unregisters the
        transport's producer.
        """
    transport = StringTransport()
    producer = object()
    encoder = ChunkedEncoder(transport)
    encoder.registerProducer(producer, True)
    self.assertIdentical(transport.producer, producer)
    self.assertTrue(transport.streaming)
    encoder.unregisterProducer()
    self.assertIdentical(transport.producer, None)
    self.assertEqual(transport.value(), b'0\r\n\r\n')