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
class SplitTests(TestCase):
    """
    Checks splitting of NMEA sentences.
    """

    def test_withChecksum(self) -> None:
        """
        An NMEA sentence with a checksum gets split correctly.
        """
        splitSentence = nmea._split(b'$GPGGA,spam,eggs*00')
        self.assertEqual(splitSentence, [b'GPGGA', b'spam', b'eggs'])

    def test_noCheckum(self) -> None:
        """
        An NMEA sentence without a checksum gets split correctly.
        """
        splitSentence = nmea._split(b'$GPGGA,spam,eggs*')
        self.assertEqual(splitSentence, [b'GPGGA', b'spam', b'eggs'])