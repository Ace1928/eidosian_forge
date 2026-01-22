import numpy as np
from holoviews import Bars, Curve, Dimension, Histogram, Points, Scatter, VectorField
from holoviews.element.comparison import ComparisonTestCase
class PointsComparisonTest(ComparisonTestCase):

    def setUp(self):
        """Variations on the constructors in the Elements notebook"""
        self.points1 = Points([(1, i) for i in range(20)])
        self.points2 = Points([(1, i) for i in range(21)])
        self.points3 = Points([(1, i * 2) for i in range(20)])

    def test_points_equal_1(self):
        self.assertEqual(self.points1, self.points1)

    def test_points_equal_2(self):
        self.assertEqual(self.points2, self.points2)

    def test_points_equal_3(self):
        self.assertEqual(self.points3, self.points3)

    def test_points_unequal_data_shape(self):
        try:
            self.assertEqual(self.points1, self.points2)
        except AssertionError as e:
            if not str(e).startswith('Points not of matching length, 20 vs. 21.'):
                raise self.failureException('Points count mismatch error not raised.')

    def test_points_unequal_data_values(self):
        try:
            self.assertEqual(self.points1, self.points3)
        except AssertionError as e:
            if not str(e).startswith('Points not almost equal to 6 decimals'):
                raise self.failureException('Points data mismatch error not raised.')