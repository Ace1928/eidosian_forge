from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
def test_zeroHeadingEdgeCase(self) -> None:
    """
        Headings can be instantiated with a value of 0 and no variation.
        """
    base.Heading(0)