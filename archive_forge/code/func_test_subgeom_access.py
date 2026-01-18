import numpy as np
import pytest
from shapely import MultiPolygon, Polygon
from shapely.geometry.base import dump_coords
from shapely.tests.geometry.test_multi import MultiGeometryTestCase
def test_subgeom_access(self):
    poly0 = Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)])
    poly1 = Polygon([(0.25, 0.25), (0.25, 0.5), (0.5, 0.5), (0.5, 0.25)])
    self.subgeom_access_test(MultiPolygon, [poly0, poly1])