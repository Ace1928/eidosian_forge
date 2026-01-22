import unittest
import numpy as np
from shapely.geometry import box, MultiPolygon, Point
def assertContainsResults(self, geom, x, y):
    from shapely.vectorized import contains
    result = contains(geom, x, y)
    x = np.asanyarray(x)
    y = np.asanyarray(y)
    self.assertIsInstance(result, np.ndarray)
    self.assertEqual(result.dtype, bool)
    result_flat = result.flat
    x_flat, y_flat = (x.flat, y.flat)
    for idx in range(x.size):
        assert result_flat[idx] == geom.contains(Point(x_flat[idx], y_flat[idx]))
    return result