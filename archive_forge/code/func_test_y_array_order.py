import unittest
import numpy as np
from shapely.geometry import box, MultiPolygon, Point
def test_y_array_order(self):
    y, x = np.mgrid[-10:10:5j, -5:15:5j]
    y = y.copy('f')
    self.assertContainsResults(self.construct_torus(), x, y)