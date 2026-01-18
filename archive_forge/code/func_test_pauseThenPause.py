from __future__ import annotations
from typing import Literal
from zope.interface import implementer
from twisted.internet.interfaces import IPushProducer
from twisted.internet.protocol import Protocol
from twisted.internet.task import Clock
from twisted.test.iosim import FakeTransport, connect, connectedServerAndClient
from twisted.trial.unittest import TestCase
def test_pauseThenPause(self) -> None:
    """
        L{StrictPushProducer.pauseProducing} raises L{ValueError} if called on a
        paused producer.
        """
    producer = self._paused()
    self.assertRaises(ValueError, producer.pauseProducing)