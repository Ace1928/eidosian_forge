import numpy as np
import holoviews as hv
from holoviews.element.comparison import ComparisonTestCase
class TestEllipsisCharts(ComparisonTestCase):

    def test_curve_ellipsis_slice_x(self):
        sliced = hv.Curve([(i, 2 * i) for i in range(10)])[2:7, ...]
        self.assertEqual(sliced.range('x'), (2, 6))

    def test_curve_ellipsis_slice_y(self):
        sliced = hv.Curve([(i, 2 * i) for i in range(10)])[..., 3:9]
        self.assertEqual(sliced.range('y'), (4, 8))

    def test_points_ellipsis_slice_x(self):
        sliced = hv.Points([(i, 2 * i) for i in range(10)])[2:7, ...]
        self.assertEqual(sliced.range('x'), (2, 6))

    def test_scatter_ellipsis_value(self):
        hv.Scatter(range(10))[..., 'y']

    def test_scatter_ellipsis_value_missing(self):
        try:
            hv.Scatter(range(10))[..., 'Non-existent']
        except Exception as e:
            if str(e) != "'Non-existent' is not an available value dimension":
                raise AssertionError('Incorrect exception raised.')

    def test_points_ellipsis_slice_y(self):
        sliced = hv.Points([(i, 2 * i) for i in range(10)])[..., 3:9]
        self.assertEqual(sliced.range('y'), (4, 8))

    def test_histogram_ellipsis_slice_value(self):
        frequencies, edges = np.histogram(range(20), 20)
        sliced = hv.Histogram((frequencies, edges))[..., 'Frequency']
        self.assertEqual(len(sliced.dimension_values(0)), 20)

    def test_histogram_ellipsis_slice_range(self):
        frequencies, edges = np.histogram(range(20), 20)
        sliced = hv.Histogram((edges, frequencies))[0:5, ...]
        self.assertEqual(len(sliced.dimension_values(0)), 5)

    def test_histogram_ellipsis_slice_value_missing(self):
        frequencies, edges = np.histogram(range(20), 20)
        with self.assertRaises(IndexError):
            hv.Histogram((frequencies, edges))[..., 'Non-existent']