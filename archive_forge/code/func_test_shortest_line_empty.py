import numpy as np
import pytest
import shapely
from shapely import GeometryCollection, LinearRing, LineString, MultiLineString, Point
from shapely.errors import UnsupportedGEOSVersionError
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.parametrize('prepare', [True, False])
def test_shortest_line_empty(prepare):
    g1 = _prepare_input(line_string, prepare)
    assert shapely.shortest_line(g1, empty_line_string) is None
    g1_empty = _prepare_input(empty_line_string, prepare)
    assert shapely.shortest_line(g1_empty, line_string) is None
    assert shapely.shortest_line(g1_empty, empty_line_string) is None