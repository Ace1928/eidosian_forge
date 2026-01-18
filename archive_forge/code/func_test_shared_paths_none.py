import numpy as np
import pytest
import shapely
from shapely import GeometryCollection, LinearRing, LineString, MultiLineString, Point
from shapely.errors import UnsupportedGEOSVersionError
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
def test_shared_paths_none():
    assert shapely.shared_paths(line_string, None) is None
    assert shapely.shared_paths(None, line_string) is None
    assert shapely.shared_paths(None, None) is None