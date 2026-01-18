from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
def test_unknownType(self) -> None:
    """
        The repr of an angle of unknown type but a given value displays that
        type and value in its repr.
        """
    a = base.Angle(1.0)
    self.assertEqual('<Angle of unknown type (1.0 degrees)>', repr(a))