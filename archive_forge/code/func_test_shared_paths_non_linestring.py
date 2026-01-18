import numpy as np
import pytest
import shapely
from shapely import GeometryCollection, LinearRing, LineString, MultiLineString, Point
from shapely.errors import UnsupportedGEOSVersionError
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
def test_shared_paths_non_linestring():
    g1 = shapely.linestrings([(0, 0), (1, 0), (1, 1)])
    g2 = shapely.points(0, 1)
    with pytest.raises(shapely.GEOSException):
        shapely.shared_paths(g1, g2)