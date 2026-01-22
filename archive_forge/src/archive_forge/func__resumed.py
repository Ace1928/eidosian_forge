from __future__ import annotations
from typing import Literal
from zope.interface import implementer
from twisted.internet.interfaces import IPushProducer
from twisted.internet.protocol import Protocol
from twisted.internet.task import Clock
from twisted.test.iosim import FakeTransport, connect, connectedServerAndClient
from twisted.trial.unittest import TestCase
def _resumed(self) -> StrictPushProducer:
    """
        @return: A new L{StrictPushProducer} which has been paused and resumed.
        """
    producer = StrictPushProducer()
    producer.pauseProducing()
    producer.resumeProducing()
    return producer