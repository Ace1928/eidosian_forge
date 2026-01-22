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
class TimestampFixerTests(FixerTestMixin, TestCase):
    """
    Tests conversion from NMEA timestamps to C{datetime.time} objects.
    """

    def test_simple(self) -> None:
        """
        A simple timestamp is converted correctly.
        """
        data = {'timestamp': '123456'}
        expected: _State = {'_time': datetime.time(12, 34, 56)}
        self._fixerTest(data, expected)

    def test_broken(self) -> None:
        """
        A broken timestamp raises C{ValueError}.
        """
        badTimestamps = ('993456', '129956', '123499')
        for t in badTimestamps:
            self._fixerTest({'timestamp': t}, exceptionClass=ValueError)