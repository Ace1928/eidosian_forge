import numpy as np
from holoviews import Dataset, HoloMap
from holoviews.core import Dimension
from holoviews.core.ndmapping import (
from holoviews.core.overlay import Overlay
from holoviews.element import Curve
from holoviews.element.comparison import ComparisonTestCase
class HoloMapTest(ComparisonTestCase):

    def setUp(self):
        self.xs = range(11)
        self.y_ints = [i * 2 for i in range(11)]
        self.ys = np.linspace(0, 1, 11)
        self.columns = Dataset(np.column_stack([self.xs, self.y_ints]), kdims=['x'], vdims=['y'])

    def test_holomap_redim(self):
        hmap = HoloMap({i: Dataset({'x': self.xs, 'y': self.ys * i}, kdims=['x'], vdims=['y']) for i in range(10)}, kdims=['z'])
        redimmed = hmap.redim(x='Time')
        self.assertEqual(redimmed.dimensions('all', True), ['z', 'Time', 'y'])

    def test_holomap_redim_nested(self):
        hmap = HoloMap({i: Dataset({'x': self.xs, 'y': self.ys * i}, kdims=['x'], vdims=['y']) for i in range(10)}, kdims=['z'])
        redimmed = hmap.redim(x='Time', z='Magnitude')
        self.assertEqual(redimmed.dimensions('all', True), ['Magnitude', 'Time', 'y'])

    def test_columns_collapse_heterogeneous(self):
        collapsed = HoloMap({i: Dataset({'x': self.xs, 'y': self.ys * i}, kdims=['x'], vdims=['y']) for i in range(10)}, kdims=['z']).collapse('z', np.mean)
        expected = Dataset({'x': self.xs, 'y': self.ys * 4.5}, kdims=['x'], vdims=['y'])
        self.compare_dataset(collapsed, expected)

    def test_columns_sample_homogeneous(self):
        samples = self.columns.sample([0, 5, 10]).dimension_values('y')
        self.assertEqual(samples, np.array([0, 10, 20]))

    def test_holomap_map_with_none(self):
        hmap = HoloMap({i: Dataset({'x': self.xs, 'y': self.ys * i}, kdims=['x'], vdims=['y']) for i in range(10)}, kdims=['z'])
        mapped = hmap.map(lambda x: x if x.range(1)[1] > 0 else None, Dataset)
        self.assertEqual(hmap[1:10], mapped)

    def test_holomap_hist_two_dims(self):
        hmap = HoloMap({i: Dataset({'x': self.xs, 'y': self.ys * i}, kdims=['x'], vdims=['y']) for i in range(10)}, kdims=['z'])
        hists = hmap.hist(dimension=['x', 'y'])
        self.assertEqual(hists['right'].last.kdims, ['y'])
        self.assertEqual(hists['top'].last.kdims, ['x'])

    def test_holomap_collapse_overlay_no_function(self):
        hmap = HoloMap({(1, 0): Curve(np.arange(8)) * Curve(-np.arange(8)), (2, 0): Curve(np.arange(8) ** 2) * Curve(-np.arange(8) ** 3)}, kdims=['A', 'B'])
        self.assertEqual(hmap.collapse(), Overlay([(('Curve', 'I'), Dataset({'A': np.concatenate([np.ones(8), np.ones(8) * 2]), 'B': np.zeros(16), 'x': np.tile(np.arange(8), 2), 'y': np.concatenate([np.arange(8), np.arange(8) ** 2])}, kdims=['A', 'B', 'x'], vdims=['y'])), (('Curve', 'II'), Dataset({'A': np.concatenate([np.ones(8), np.ones(8) * 2]), 'B': np.zeros(16), 'x': np.tile(np.arange(8), 2), 'y': np.concatenate([-np.arange(8), -np.arange(8) ** 3])}, kdims=['A', 'B', 'x'], vdims=['y']))]))

    def test_holomap_collapse_overlay_max(self):
        hmap = HoloMap({(1, 0): Curve(np.arange(8)) * Curve(-np.arange(8)), (2, 0): Curve(np.arange(8) ** 2) * Curve(-np.arange(8) ** 3)}, kdims=['A', 'B'])
        self.assertEqual(hmap.collapse(function=np.max), Overlay([(('Curve', 'I'), Curve({'x': np.arange(8), 'y': np.arange(8) ** 2}, kdims=['x'], vdims=['y'])), (('Curve', 'II'), Curve({'x': np.arange(8), 'y': -np.arange(8)}, kdims=['x'], vdims=['y']))]))