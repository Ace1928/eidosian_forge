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
def test_unregisterProducer(self) -> None:
    """
        L{StringTransport.unregisterProducer} causes the transport to forget
        about the registered producer and makes it possible to register a new
        one.
        """
    oldProducer = object()
    newProducer = object()
    self.transport.registerProducer(oldProducer, False)
    self.transport.unregisterProducer()
    self.assertIsNone(self.transport.producer)
    self.transport.registerProducer(newProducer, True)
    self.assertIs(self.transport.producer, newProducer)
    self.assertTrue(self.transport.streaming)