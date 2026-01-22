from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
class HeadingTests(TestCase):
    """
    Tests for the L{twisted.positioning.base.Heading} class.
    """

    def test_simple(self) -> None:
        """
        Tests that a simple heading has a value in decimal degrees, which is
        also its value when converted to a float. Its variation, and by
        consequence its corrected heading, is L{None}.
        """
        h = base.Heading(1.0)
        self.assertEqual(h.inDecimalDegrees, 1.0)
        self.assertEqual(float(h), 1.0)
        self.assertIsNone(h.variation)
        self.assertIsNone(h.correctedHeading)

    def test_headingWithoutVariationRepr(self) -> None:
        """
        A repr of a heading with no variation reports its value and that the
        variation is unknown.
        """
        heading = base.Heading(1.0)
        expectedRepr = '<Heading (1.0 degrees, unknown variation)>'
        self.assertEqual(repr(heading), expectedRepr)

    def test_headingWithVariationRepr(self) -> None:
        """
        A repr of a heading with known variation reports its value and the
        value of that variation.
        """
        angle, variation = (1.0, -10.0)
        heading = base.Heading.fromFloats(angle, variationValue=variation)
        reprTemplate = '<Heading ({0} degrees, <Variation ({1} degrees)>)>'
        self.assertEqual(repr(heading), reprTemplate.format(angle, variation))

    def test_valueEquality(self) -> None:
        """
        Headings with the same values compare equal.
        """
        self.assertEqual(base.Heading(1.0), base.Heading(1.0))

    def test_valueInequality(self) -> None:
        """
        Headings with different values compare unequal.
        """
        self.assertNotEqual(base.Heading(1.0), base.Heading(2.0))

    def test_zeroHeadingEdgeCase(self) -> None:
        """
        Headings can be instantiated with a value of 0 and no variation.
        """
        base.Heading(0)

    def test_zeroHeading180DegreeVariationEdgeCase(self) -> None:
        """
        Headings can be instantiated with a value of 0 and a variation of 180
        degrees.
        """
        base.Heading(0, 180)

    def _badValueTest(self, **kw: float) -> None:
        """
        Helper function for verifying that bad values raise C{ValueError}.

        @param kw: The keyword arguments passed to L{base.Heading.fromFloats}.
        """
        self.assertRaises(ValueError, base.Heading.fromFloats, **kw)

    def test_badAngleValueEdgeCase(self) -> None:
        """
        Headings can not be instantiated with a value of 360 degrees.
        """
        self._badValueTest(angleValue=360.0)

    def test_badVariationEdgeCase(self) -> None:
        """
        Headings can not be instantiated with a variation of -180 degrees.
        """
        self._badValueTest(variationValue=-180.0)

    def test_negativeHeading(self) -> None:
        """
        Negative heading values raise C{ValueError}.
        """
        self._badValueTest(angleValue=-10.0)

    def test_headingTooLarge(self) -> None:
        """
        Heading values greater than C{360.0} raise C{ValueError}.
        """
        self._badValueTest(angleValue=370.0)

    def test_variationTooNegative(self) -> None:
        """
        Variation values less than C{-180.0} raise C{ValueError}.
        """
        self._badValueTest(variationValue=-190.0)

    def test_variationTooPositive(self) -> None:
        """
        Variation values greater than C{180.0} raise C{ValueError}.
        """
        self._badValueTest(variationValue=190.0)

    def test_correctedHeading(self) -> None:
        """
        A heading with a value and a variation has a corrected heading.
        """
        h = base.Heading.fromFloats(1.0, variationValue=-10.0)
        self.assertEqual(h.correctedHeading, base.Angle(11.0, Angles.HEADING))

    def test_correctedHeadingOverflow(self) -> None:
        """
        A heading with a value and a variation has the appropriate corrected
        heading value, even when the variation puts it across the 360 degree
        boundary.
        """
        h = base.Heading.fromFloats(359.0, variationValue=-2.0)
        self.assertEqual(h.correctedHeading, base.Angle(1.0, Angles.HEADING))

    def test_correctedHeadingOverflowEdgeCase(self) -> None:
        """
        A heading with a value and a variation has the appropriate corrected
        heading value, even when the variation puts it exactly at the 360
        degree boundary.
        """
        h = base.Heading.fromFloats(359.0, variationValue=-1.0)
        self.assertEqual(h.correctedHeading, base.Angle(0.0, Angles.HEADING))

    def test_correctedHeadingUnderflow(self) -> None:
        """
        A heading with a value and a variation has the appropriate corrected
        heading value, even when the variation puts it under the 0 degree
        boundary.
        """
        h = base.Heading.fromFloats(1.0, variationValue=2.0)
        self.assertEqual(h.correctedHeading, base.Angle(359.0, Angles.HEADING))

    def test_correctedHeadingUnderflowEdgeCase(self) -> None:
        """
        A heading with a value and a variation has the appropriate corrected
        heading value, even when the variation puts it exactly at the 0
        degree boundary.
        """
        h = base.Heading.fromFloats(1.0, variationValue=1.0)
        self.assertEqual(h.correctedHeading, base.Angle(0.0, Angles.HEADING))

    def test_setVariationSign(self) -> None:
        """
        Setting the sign of a heading changes the variation sign.
        """
        h = base.Heading.fromFloats(1.0, variationValue=1.0)
        h.setSign(1)
        self.assertEqual(h.variation.inDecimalDegrees, 1.0)
        h.setSign(-1)
        self.assertEqual(h.variation.inDecimalDegrees, -1.0)

    def test_setBadVariationSign(self) -> None:
        """
        Setting the sign of a heading to values that aren't C{-1} or C{1}
        raises C{ValueError} and does not affect the heading.
        """
        h = base.Heading.fromFloats(1.0, variationValue=1.0)
        self.assertRaises(ValueError, h.setSign, -50)
        self.assertEqual(h.variation.inDecimalDegrees, 1.0)
        self.assertRaises(ValueError, h.setSign, 0)
        self.assertEqual(h.variation.inDecimalDegrees, 1.0)
        self.assertRaises(ValueError, h.setSign, 50)
        self.assertEqual(h.variation.inDecimalDegrees, 1.0)

    def test_setUnknownVariationSign(self) -> None:
        """
        Setting the sign on a heading with unknown variation raises
        C{ValueError}.
        """
        h = base.Heading.fromFloats(1.0)
        self.assertIsNone(h.variation.inDecimalDegrees)
        self.assertRaises(ValueError, h.setSign, 1)