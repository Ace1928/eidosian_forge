import warnings
import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, MultiPolygon, Point, Polygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import all_types
from shapely.tests.common import empty as empty_geometry_collection
from shapely.tests.common import (
@pytest.mark.skipif(shapely.geos_version < (3, 6, 0), reason='GEOS < 3.6')
def test_set_precision_intersection():
    """Operations should use the most precise presision grid size of the inputs"""
    box1 = shapely.normalize(shapely.box(0, 0, 0.9, 0.9))
    box2 = shapely.normalize(shapely.box(0.75, 0, 1.75, 0.75))
    assert shapely.get_precision(shapely.intersection(box1, box2)) == 0
    box1 = shapely.set_precision(box1, 0.5)
    box2 = shapely.set_precision(box2, 1)
    out = shapely.intersection(box1, box2)
    assert shapely.get_precision(out) == 0.5
    assert_geometries_equal(out, LineString([(1, 1), (1, 0)]))