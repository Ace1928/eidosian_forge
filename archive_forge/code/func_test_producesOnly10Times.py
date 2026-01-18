from __future__ import annotations
from typing import Callable
from zope.interface.verify import verifyObject
from typing_extensions import Protocol
from twisted.internet.address import IPv4Address
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, Factory
from twisted.internet.testing import (
from twisted.python.reflect import namedAny
from twisted.trial.unittest import TestCase
def test_producesOnly10Times(self) -> None:
    """
        When the L{NonStreamingProducer} has resumeProducing called 10 times,
        it writes the counter each time and then fails.
        """
    consumer = TestConsumer()
    producer = NonStreamingProducer(consumer)
    consumer.registerProducer(producer, False)
    self.assertIs(consumer.producer, producer)
    self.assertIs(producer.consumer, consumer)
    self.assertFalse(consumer.producerStreaming)
    for _ in range(10):
        producer.resumeProducing()
    expectedWrites = [b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8', b'9']
    self.assertIsNone(consumer.producer)
    self.assertIsNone(consumer.producerStreaming)
    self.assertIsNone(producer.consumer)
    self.assertEqual(consumer.writes, expectedWrites)
    self.assertRaises(RuntimeError, producer.resumeProducing)