from __future__ import annotations
from typing import Literal
from zope.interface import implementer
from twisted.internet.interfaces import IPushProducer
from twisted.internet.protocol import Protocol
from twisted.internet.task import Clock
from twisted.test.iosim import FakeTransport, connect, connectedServerAndClient
from twisted.trial.unittest import TestCase
def test_timeAdvances(self) -> None:
    """
        L{IOPump.pump} advances time in the given L{Clock}.
        """
    time_passed = []
    clock = Clock()
    _, _, pump = connectedServerAndClient(Protocol, Protocol, clock=clock)
    clock.callLater(0, lambda: time_passed.append(True))
    self.assertFalse(time_passed)
    pump.pump()
    self.assertTrue(time_passed)