from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
def test_positiveLatitude(self) -> None:
    """
        Positive latitudes have a repr that specifies their type and value.
        """
    coordinate = base.Coordinate(10.0, Angles.LATITUDE)
    expectedRepr = f'<Latitude ({10.0} degrees)>'
    self.assertEqual(repr(coordinate), expectedRepr)