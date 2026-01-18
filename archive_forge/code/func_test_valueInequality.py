from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
def test_valueInequality(self) -> None:
    """
        Headings with different values compare unequal.
        """
    self.assertNotEqual(base.Heading(1.0), base.Heading(2.0))