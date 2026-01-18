from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
def test_setDOPWithInvariant(self) -> None:
    """
        Attempting to set the PDOP value to value inconsisted with HDOP and
        VDOP when checking the invariant raises C{ValueError}.
        """
    pe = base.PositionError(hdop=1.0, vdop=1.0, testInvariant=True)
    pdop = pe.pdop

    def setPDOP(pe: base.PositionError) -> None:
        pe.pdop = 100.0
    self.assertRaises(ValueError, setPDOP, pe)
    self.assertEqual(pe.pdop, pdop)