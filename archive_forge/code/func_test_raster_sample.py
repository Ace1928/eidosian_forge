import numpy as np
from holoviews.element import HSV, RGB, Curve, Image, QuadMesh, Raster
from holoviews.element.comparison import ComparisonTestCase
def test_raster_sample(self):
    raster = Raster(self.array1)
    self.assertEqual(raster.sample(y=0), Curve(np.array([(0, 0), (1, 1), (2, 2)]), kdims=['x'], vdims=['z']))