import unittest
import pytest
from shapely.errors import GeometryTypeError
from shapely.geometry import (
from shapely.ops import linemerge, split, unary_union
def test_split_poly_with_other(self):
    with pytest.raises(GeometryTypeError):
        split(self.poly_simple, Point(1, 1))
    with pytest.raises(GeometryTypeError):
        split(self.poly_simple, MultiPoint([(1, 1), (3, 4)]))
    with pytest.raises(GeometryTypeError):
        split(self.poly_simple, self.poly_hole)