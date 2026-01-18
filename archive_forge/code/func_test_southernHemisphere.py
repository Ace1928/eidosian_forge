from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
def test_southernHemisphere(self) -> None:
    """
        Negative latitudes are in the southern hemisphere.
        """
    coordinate = base.Coordinate(-1.0, Angles.LATITUDE)
    self.assertEqual(coordinate.hemisphere, Directions.SOUTH)