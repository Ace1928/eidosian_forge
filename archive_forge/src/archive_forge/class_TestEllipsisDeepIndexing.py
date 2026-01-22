import numpy as np
import holoviews as hv
from holoviews.element.comparison import ComparisonTestCase
class TestEllipsisDeepIndexing(ComparisonTestCase):

    def test_deep_ellipsis_curve_slicing_1(self):
        hmap = hv.HoloMap({i: hv.Curve([(j, j) for j in range(10)]) for i in range(10)})
        sliced = hmap[2:5, ...]
        self.assertEqual(sliced.keys(), [2, 3, 4])

    def test_deep_ellipsis_curve_slicing_2(self):
        hmap = hv.HoloMap({i: hv.Curve([(j, j) for j in range(10)]) for i in range(10)})
        sliced = hmap[2:5, 1:8, ...]
        self.assertEqual(sliced.last.range('x'), (1, 7))

    def test_deep_ellipsis_curve_slicing_3(self):
        hmap = hv.HoloMap({i: hv.Curve([(j, 2 * j) for j in range(10)]) for i in range(10)})
        sliced = hmap[..., 2:5]
        self.assertEqual(sliced.last.range('y'), (2, 4))