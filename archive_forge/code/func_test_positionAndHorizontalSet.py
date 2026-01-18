from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
def test_positionAndHorizontalSet(self) -> None:
    """
        The VDOP is correctly determined from PDOP and HDOP.
        """
    pdop, hdop = (2.0, 1.0)
    vdop = (pdop ** 2 - hdop ** 2) ** 0.5
    pe = base.PositionError(pdop=pdop, hdop=hdop)
    self._testDOP(pe, pdop, hdop, vdop)