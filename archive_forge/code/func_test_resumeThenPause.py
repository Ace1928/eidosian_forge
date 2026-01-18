from __future__ import annotations
from typing import Literal
from zope.interface import implementer
from twisted.internet.interfaces import IPushProducer
from twisted.internet.protocol import Protocol
from twisted.internet.task import Clock
from twisted.test.iosim import FakeTransport, connect, connectedServerAndClient
from twisted.trial.unittest import TestCase
def test_resumeThenPause(self) -> None:
    """
        L{StrictPushProducer} is paused if C{pauseProducing} is called on a
        resumed producer.
        """
    producer = self._resumed()
    producer.pauseProducing()
    self.assertPaused(producer)