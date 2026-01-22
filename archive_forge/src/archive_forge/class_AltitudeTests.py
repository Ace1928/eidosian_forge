from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
class AltitudeTests(TestCase):
    """
    Tests for the L{twisted.positioning.base.Altitude} class.
    """

    def test_value(self) -> None:
        """
        Altitudes can be instantiated and reports the correct value in
        meters and feet, as well as when converted to float.
        """
        altitude = base.Altitude(1.0)
        self.assertEqual(float(altitude), 1.0)
        self.assertEqual(altitude.inMeters, 1.0)
        self.assertEqual(altitude.inFeet, 1.0 / base.METERS_PER_FOOT)

    def test_repr(self) -> None:
        """
        Altitudes report their type and value in their repr.
        """
        altitude = base.Altitude(1.0)
        self.assertEqual(repr(altitude), '<Altitude (1.0 m)>')

    def test_equality(self) -> None:
        """
        Altitudes with equal values compare equal.
        """
        firstAltitude = base.Altitude(1.0)
        secondAltitude = base.Altitude(1.0)
        self.assertEqual(firstAltitude, secondAltitude)

    def test_inequality(self) -> None:
        """
        Altitudes with different values don't compare equal.
        """
        firstAltitude = base.Altitude(1.0)
        secondAltitude = base.Altitude(-1.0)
        self.assertNotEqual(firstAltitude, secondAltitude)