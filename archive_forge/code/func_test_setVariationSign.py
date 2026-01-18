from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
def test_setVariationSign(self) -> None:
    """
        Setting the sign of a heading changes the variation sign.
        """
    h = base.Heading.fromFloats(1.0, variationValue=1.0)
    h.setSign(1)
    self.assertEqual(h.variation.inDecimalDegrees, 1.0)
    h.setSign(-1)
    self.assertEqual(h.variation.inDecimalDegrees, -1.0)