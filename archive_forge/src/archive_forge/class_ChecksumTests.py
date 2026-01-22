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
class ChecksumTests(TestCase):
    """
    NMEA sentence checksum verification tests.
    """

    def test_valid(self) -> None:
        """
        Sentences with valid checksums get validated.
        """
        nmea._validateChecksum(GPGGA)

    def test_missing(self) -> None:
        """
        Sentences with missing checksums get validated.
        """
        nmea._validateChecksum(GPGGA[:-2])

    def test_invalid(self) -> None:
        """
        Sentences with a bad checksum raise L{base.InvalidChecksum} when
        attempting to validate them.
        """
        validate = nmea._validateChecksum
        bareSentence, checksum = GPGGA.split(b'*')
        badChecksum = b'%d' % (int(checksum, 16) + 1,)
        sentences = [bareSentence + b'*' + badChecksum]
        for s in sentences:
            self.assertRaises(base.InvalidChecksum, validate, s)