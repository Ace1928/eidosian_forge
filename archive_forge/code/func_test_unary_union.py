import random
import unittest
from functools import partial
from itertools import islice
import pytest
from shapely.errors import ShapelyDeprecationWarning
from shapely.geometry import MultiPolygon, Point
from shapely.ops import cascaded_union, unary_union
def test_unary_union(self):
    patches = [Point(xy).buffer(0.05) for xy in self.coords]
    u = unary_union(patches)
    assert u.geom_type == 'MultiPolygon'
    assert u.area == pytest.approx(0.718572540569)