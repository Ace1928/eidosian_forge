from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
class PositionErrorTests(TestCase):
    """
    Tests for L{twisted.positioning.base.PositionError}.
    """

    def test_allUnset(self) -> None:
        """
        In an empty L{base.PositionError} with no invariant testing, all
        dilutions of positions are L{None}.
        """
        positionError = base.PositionError()
        self.assertIsNone(positionError.pdop)
        self.assertIsNone(positionError.hdop)
        self.assertIsNone(positionError.vdop)

    def test_allUnsetWithInvariant(self) -> None:
        """
        In an empty L{base.PositionError} with invariant testing, all
        dilutions of positions are L{None}.
        """
        positionError = base.PositionError(testInvariant=True)
        self.assertIsNone(positionError.pdop)
        self.assertIsNone(positionError.hdop)
        self.assertIsNone(positionError.vdop)

    def test_withoutInvariant(self) -> None:
        """
        L{base.PositionError}s can be instantiated with just a HDOP.
        """
        positionError = base.PositionError(hdop=1.0)
        self.assertEqual(positionError.hdop, 1.0)

    def test_withInvariant(self) -> None:
        """
        Creating a simple L{base.PositionError} with just a HDOP while
        checking the invariant works.
        """
        positionError = base.PositionError(hdop=1.0, testInvariant=True)
        self.assertEqual(positionError.hdop, 1.0)

    def test_invalidWithoutInvariant(self) -> None:
        """
        Creating a L{base.PositionError} with values set to an impossible
        combination works if the invariant is not checked.
        """
        error = base.PositionError(pdop=1.0, vdop=1.0, hdop=1.0)
        self.assertEqual(error.pdop, 1.0)
        self.assertEqual(error.hdop, 1.0)
        self.assertEqual(error.vdop, 1.0)

    def test_invalidWithInvariant(self) -> None:
        """
        Creating a L{base.PositionError} with values set to an impossible
        combination raises C{ValueError} if the invariant is being tested.
        """
        self.assertRaises(ValueError, base.PositionError, pdop=1.0, vdop=1.0, hdop=1.0, testInvariant=True)

    def test_setDOPWithoutInvariant(self) -> None:
        """
        You can set the PDOP value to value inconsisted with HDOP and VDOP
        when not checking the invariant.
        """
        pe = base.PositionError(hdop=1.0, vdop=1.0)
        pe.pdop = 100.0
        self.assertEqual(pe.pdop, 100.0)

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
    REPR_TEMPLATE = '<PositionError (pdop: %s, hdop: %s, vdop: %s)>'

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

    def test_positionAndHorizontalSet(self) -> None:
        """
        The VDOP is correctly determined from PDOP and HDOP.
        """
        pdop, hdop = (2.0, 1.0)
        vdop = (pdop ** 2 - hdop ** 2) ** 0.5
        pe = base.PositionError(pdop=pdop, hdop=hdop)
        self._testDOP(pe, pdop, hdop, vdop)

    def test_positionAndVerticalSet(self) -> None:
        """
        The HDOP is correctly determined from PDOP and VDOP.
        """
        pdop, vdop = (2.0, 1.0)
        hdop = (pdop ** 2 - vdop ** 2) ** 0.5
        pe = base.PositionError(pdop=pdop, vdop=vdop)
        self._testDOP(pe, pdop, hdop, vdop)

    def test_horizontalAndVerticalSet(self) -> None:
        """
        The PDOP is correctly determined from HDOP and VDOP.
        """
        hdop, vdop = (1.0, 1.0)
        pdop = (hdop ** 2 + vdop ** 2) ** 0.5
        pe = base.PositionError(hdop=hdop, vdop=vdop)
        self._testDOP(pe, pdop, hdop, vdop)