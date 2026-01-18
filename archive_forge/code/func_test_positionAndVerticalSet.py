from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
def test_positionAndVerticalSet(self) -> None:
    """
        The HDOP is correctly determined from PDOP and VDOP.
        """
    pdop, vdop = (2.0, 1.0)
    hdop = (pdop ** 2 - vdop ** 2) ** 0.5
    pe = base.PositionError(pdop=pdop, vdop=vdop)
    self._testDOP(pe, pdop, hdop, vdop)