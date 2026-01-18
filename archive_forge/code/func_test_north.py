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
def test_north(self) -> None:
    """
        NMEA coordinate representations in the northern hemisphere
        convert correctly.
        """
    sentenceData = {'latitudeFloat': '1030.000', 'latitudeHemisphere': 'N'}
    state: _State = {'latitude': base.Coordinate(10.5, Angles.LATITUDE)}
    self._fixerTest(sentenceData, state)