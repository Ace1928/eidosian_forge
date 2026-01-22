from __future__ import annotations
import datetime
from operator import attrgetter
from typing import Callable, Iterable, TypedDict
from zope.interface import implementer
from constantly import NamedConstant
from typing_extensions import Literal, Protocol
from twisted.positioning import base, ipositioning, nmea
from twisted.positioning.base import Angles
from twisted.positioning.test.receiver import MockPositioningReceiver
from twisted.trial.unittest import TestCase
@implementer(ipositioning.INMEAReceiver)
class NMEATestReceiver:
    """
    An NMEA receiver for testing.

    Remembers the last sentence it has received.
    """

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        """
        Forgets the received sentence (if any), by setting
        C{self.receivedSentence} to L{None}.
        """
        self.receivedSentence: nmea.NMEASentence | None = None

    def sentenceReceived(self, sentence: nmea.NMEASentence) -> None:
        self.receivedSentence = sentence