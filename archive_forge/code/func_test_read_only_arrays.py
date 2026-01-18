import numpy as np
import pytest
from numpy.testing import assert_allclose
import shapely
from shapely import MultiLineString, MultiPoint, MultiPolygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.parametrize('geom', all_types)
def test_read_only_arrays(geom):
    typ, coords, offsets = shapely.to_ragged_array([geom, geom])
    coords.flags.writeable = False
    for arr in offsets:
        arr.flags.writeable = False
    result = shapely.from_ragged_array(typ, coords, offsets)
    assert_geometries_equal(result, [geom, geom])