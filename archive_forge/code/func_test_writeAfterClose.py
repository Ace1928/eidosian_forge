from __future__ import annotations
from typing import Literal
from zope.interface import implementer
from twisted.internet.interfaces import IPushProducer
from twisted.internet.protocol import Protocol
from twisted.internet.task import Clock
from twisted.test.iosim import FakeTransport, connect, connectedServerAndClient
from twisted.trial.unittest import TestCase
def test_writeAfterClose(self) -> None:
    """
        L{FakeTransport.write} will accept writes after transport was closed,
        but the data will be silently discarded.
        """
    a = FakeTransport(object(), False)
    a.write(b'before')
    a.loseConnection()
    a.write(b'after')
    self.assertEqual(b''.join(a.stream), b'before')