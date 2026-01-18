import numpy as np
import pytest
import shapely
from shapely import GeometryCollection, LinearRing, LineString, MultiLineString, Point
from shapely.errors import UnsupportedGEOSVersionError
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.parametrize('prepare', [True, False])
def test_shortest_line(prepare):
    g1 = shapely.linestrings([(0, 0), (1, 0), (1, 1)])
    g2 = shapely.linestrings([(0, 3), (3, 0)])
    actual = shapely.shortest_line(_prepare_input(g1, prepare), g2)
    expected = shapely.linestrings([(1, 1), (1.5, 1.5)])
    assert shapely.equals(actual, expected)