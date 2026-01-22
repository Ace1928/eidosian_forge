from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
def _testDOP(self, pe: base.PositionError, pdop: float | None, hdop: float | None, vdop: float | None) -> None:
    """
        Tests the DOP values in a position error, and the repr of that
        position error.

        @param pe: The position error under test.
        @type pe: C{PositionError}
        @param pdop: The expected position dilution of precision.
        @type pdop: C{float} or L{None}
        @param hdop: The expected horizontal dilution of precision.
        @type hdop: C{float} or L{None}
        @param vdop: The expected vertical dilution of precision.
        @type vdop: C{float} or L{None}
        """
    self.assertEqual(pe.pdop, pdop)
    self.assertEqual(pe.hdop, hdop)
    self.assertEqual(pe.vdop, vdop)
    self.assertEqual(repr(pe), self.REPR_TEMPLATE % (pdop, hdop, vdop))