import numpy as np
from holoviews.element import HSV, RGB, Curve, Image, QuadMesh, Raster
from holoviews.element.comparison import ComparisonTestCase
def test_raster_range_masked(self):
    arr = np.random.rand(10, 10) - 0.5
    arr = np.ma.masked_where(arr <= 0, arr)
    rrange = Raster(arr).range(2)
    self.assertEqual(rrange, (np.min(arr), np.max(arr)))