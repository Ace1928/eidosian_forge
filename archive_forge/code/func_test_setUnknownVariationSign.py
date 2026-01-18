from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
def test_setUnknownVariationSign(self) -> None:
    """
        Setting the sign on a heading with unknown variation raises
        C{ValueError}.
        """
    h = base.Heading.fromFloats(1.0)
    self.assertIsNone(h.variation.inDecimalDegrees)
    self.assertRaises(ValueError, h.setSign, 1)