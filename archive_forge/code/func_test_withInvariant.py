from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
def test_withInvariant(self) -> None:
    """
        Creating a simple L{base.PositionError} with just a HDOP while
        checking the invariant works.
        """
    positionError = base.PositionError(hdop=1.0, testInvariant=True)
    self.assertEqual(positionError.hdop, 1.0)