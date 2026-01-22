from holoviews import Dimension, ErrorBars, Histogram
from holoviews.element.comparison import ComparisonTestCase
class HistogramRangeTests(ComparisonTestCase):

    def test_histogram_range_x(self):
        r = Histogram(([0, 1, 2, 3], [1, 2, 3])).range(0)
        self.assertEqual(r, (0.0, 3.0))

    def test_histogram_range_x_explicit(self):
        r = Histogram(([0, 1, 2, 3], [1, 2, 3]), kdims=[Dimension('x', range=(-1, 4.0))]).range(0)
        self.assertEqual(r, (-1.0, 4.0))

    def test_histogram_range_x_explicit_upper(self):
        r = Histogram(([0, 1, 2, 3], [1, 2, 3]), kdims=[Dimension('x', range=(None, 4.0))]).range(0)
        self.assertEqual(r, (0, 4.0))

    def test_histogram_range_x_explicit_lower(self):
        r = Histogram(([0, 1, 2, 3], [1, 2, 3]), kdims=[Dimension('x', range=(-1, None))]).range(0)
        self.assertEqual(r, (-1.0, 3.0))

    def test_histogram_range_y(self):
        r = Histogram(([0, 1, 2, 3], [1, 2, 3])).range(1)
        self.assertEqual(r, (1.0, 3.0))

    def test_histogram_range_y_explicit(self):
        r = Histogram(([0, 1, 2, 3], [1, 2, 3]), vdims=[Dimension('y', range=(0, 4.0))]).range(1)
        self.assertEqual(r, (0.0, 4.0))

    def test_histogram_range_y_explicit_upper(self):
        r = Histogram(([0, 1, 2, 3], [1, 2, 3]), vdims=[Dimension('y', range=(None, 4.0))]).range(1)
        self.assertEqual(r, (1.0, 4.0))

    def test_histogram_range_y_explicit_lower(self):
        r = Histogram(([0, 1, 2, 3], [1, 2, 3]), vdims=[Dimension('y', range=(0.0, None))]).range(1)
        self.assertEqual(r, (0.0, 3.0))