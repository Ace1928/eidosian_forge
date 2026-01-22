import numpy as np
from holoviews import Bars, Curve, Dimension, Histogram, Points, Scatter, VectorField
from holoviews.element.comparison import ComparisonTestCase
class ScatterComparisonTest(ComparisonTestCase):

    def setUp(self):
        """Variations on the constructors in the Elements notebook"""
        self.scatter1 = Scatter([(1, i) for i in range(20)])
        self.scatter2 = Scatter([(1, i) for i in range(21)])
        self.scatter3 = Scatter([(1, i * 2) for i in range(20)])

    def test_scatter_equal_1(self):
        self.assertEqual(self.scatter1, self.scatter1)

    def test_scatter_equal_2(self):
        self.assertEqual(self.scatter2, self.scatter2)

    def test_scatter_equal_3(self):
        self.assertEqual(self.scatter3, self.scatter3)

    def test_scatter_unequal_data_shape(self):
        try:
            self.assertEqual(self.scatter1, self.scatter2)
        except AssertionError as e:
            if not str(e).startswith('Scatter not of matching length, 20 vs. 21.'):
                raise self.failureException('Scatter data mismatch error not raised.')

    def test_scatter_unequal_data_values(self):
        try:
            self.assertEqual(self.scatter1, self.scatter3)
        except AssertionError as e:
            if not str(e).startswith('Scatter not almost equal to 6 decimals'):
                raise self.failureException('Scatter data mismatch error not raised.')