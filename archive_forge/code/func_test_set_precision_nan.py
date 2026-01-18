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
@pytest.mark.parametrize('mode', ('valid_output', 'pointwise', 'keep_collapsed'))
def test_set_precision_nan(mode):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        actual = shapely.set_precision(line_string_nan, 1, mode=mode)
        assert_geometries_equal(actual, line_string_nan)