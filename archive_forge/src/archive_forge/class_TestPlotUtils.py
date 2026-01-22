import numpy as np
from holoviews import Dimension, NdOverlay, Overlay
from holoviews.core.options import Cycle, Store
from holoviews.core.spaces import DynamicMap, HoloMap
from holoviews.element import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import operation
from holoviews.plotting.util import (
from holoviews.streams import PointerX
class TestPlotUtils(ComparisonTestCase):

    def test_get_min_distance_float32_type(self):
        xs, ys = (np.arange(0, 2.0, 0.2, dtype='float32'), np.arange(0, 2.0, 0.2, dtype='float32'))
        X, Y = np.meshgrid(xs, ys)
        dist = get_min_distance(Points((X.flatten(), Y.flatten())))
        self.assertEqual(float(round(dist, 5)), 0.2)

    def test_get_min_distance_int32_type(self):
        xs, ys = (np.arange(0, 10, dtype='int32'), np.arange(0, 10, dtype='int32'))
        X, Y = np.meshgrid(xs, ys)
        dist = get_min_distance(Points((X.flatten(), Y.flatten())))
        self.assertEqual(dist, 1.0)

    def test_get_min_distance_float32_type_no_scipy(self):
        xs, ys = (np.arange(0, 2.0, 0.2, dtype='float32'), np.arange(0, 2.0, 0.2, dtype='float32'))
        X, Y = np.meshgrid(xs, ys)
        dist = _get_min_distance_numpy(Points((X.flatten(), Y.flatten())))
        self.assertEqual(dist, np.float32(0.2))

    def test_get_min_distance_int32_type_no_scipy(self):
        xs, ys = (np.arange(0, 10, dtype='int32'), np.arange(0, 10, dtype='int32'))
        X, Y = np.meshgrid(xs, ys)
        dist = _get_min_distance_numpy(Points((X.flatten(), Y.flatten())))
        self.assertEqual(dist, 1.0)