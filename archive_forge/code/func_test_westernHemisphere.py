from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
def test_westernHemisphere(self) -> None:
    """
        Negative longitudes are in the western hemisphere.
        """
    coordinate = base.Coordinate(-1.0, Angles.LONGITUDE)
    self.assertEqual(coordinate.hemisphere, Directions.WEST)