from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
def test_valueEquality(self) -> None:
    """
        Headings with the same values compare equal.
        """
    self.assertEqual(base.Heading(1.0), base.Heading(1.0))