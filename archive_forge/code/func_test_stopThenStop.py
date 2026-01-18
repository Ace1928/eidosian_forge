from __future__ import annotations
from typing import Literal
from zope.interface import implementer
from twisted.internet.interfaces import IPushProducer
from twisted.internet.protocol import Protocol
from twisted.internet.task import Clock
from twisted.test.iosim import FakeTransport, connect, connectedServerAndClient
from twisted.trial.unittest import TestCase
def test_stopThenStop(self) -> None:
    """
        L{StrictPushProducer.stopProducing} raises L{ValueError} if called when
        the producer is stopped.
        """
    self.assertRaises(ValueError, self._stopped().stopProducing)