import random
import unittest
from functools import partial
from itertools import islice
import pytest
from shapely.errors import ShapelyDeprecationWarning
from shapely.geometry import MultiPolygon, Point
from shapely.ops import cascaded_union, unary_union
def test_unary_union_multi(self):
    patches = MultiPolygon([Point(xy).buffer(0.05) for xy in self.coords])
    assert unary_union(patches).area == pytest.approx(0.71857254056)
    assert unary_union([patches, patches]).area == pytest.approx(0.71857254056)