import unittest
import pytest
import shapely
from shapely.geometry import Point, Polygon
def test_unary_predicates(self):
    point = Point(0.0, 0.0)
    assert not point.is_empty
    assert point.is_valid
    assert point.is_simple
    assert not point.is_ring
    assert not point.has_z