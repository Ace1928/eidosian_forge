import numpy as np
from holoviews import Bars, Curve, Dimension, Histogram, Points, Scatter, VectorField
from holoviews.element.comparison import ComparisonTestCase
class VectorFieldComparisonTest(ComparisonTestCase):

    def setUp(self):
        """Variations on the constructors in the Elements notebook"""
        x, y = np.mgrid[-10:10, -10:10] * 0.25
        sine_rings = np.sin(x ** 2 + y ** 2) * np.pi + np.pi
        exp_falloff1 = 1 / np.exp((x ** 2 + y ** 2) / 8)
        exp_falloff2 = 1 / np.exp((x ** 2 + y ** 2) / 9)
        self.vfield1 = VectorField([x, y, sine_rings, exp_falloff1])
        self.vfield2 = VectorField([x, y, sine_rings, exp_falloff2])

    def test_vfield_equal_1(self):
        self.assertEqual(self.vfield1, self.vfield1)

    def test_vfield_equal_2(self):
        self.assertEqual(self.vfield2, self.vfield2)

    def test_vfield_unequal_1(self):
        try:
            self.assertEqual(self.vfield1, self.vfield2)
        except AssertionError as e:
            if not str(e).startswith('VectorField not almost equal to 6 decimals'):
                raise self.failureException('VectorField  data mismatch error not raised.')