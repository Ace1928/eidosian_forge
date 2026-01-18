from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
def test_unknownAngleInDegreesMinutesSeconds(self) -> None:
    """
        If the vaue of a coordinate is L{None}, its values in degrees,
        minutes, seconds is also L{None}.
        """
    c = base.Coordinate(None, None)
    self.assertIsNone(c.inDegreesMinutesSeconds)