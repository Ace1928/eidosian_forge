from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
def test_setDOPWithoutInvariant(self) -> None:
    """
        You can set the PDOP value to value inconsisted with HDOP and VDOP
        when not checking the invariant.
        """
    pe = base.PositionError(hdop=1.0, vdop=1.0)
    pe.pdop = 100.0
    self.assertEqual(pe.pdop, 100.0)