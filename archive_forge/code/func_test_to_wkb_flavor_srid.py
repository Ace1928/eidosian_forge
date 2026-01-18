import json
import pickle
import struct
import warnings
import numpy as np
import pytest
import shapely
from shapely import GeometryCollection, LineString, Point, Polygon
from shapely.errors import UnsupportedGEOSVersionError
from shapely.testing import assert_geometries_equal
from shapely.tests.common import all_types, empty_point, empty_point_z, point, point_z
@pytest.mark.skipif(shapely.geos_version < (3, 10, 0), reason='GEOS < 3.10.0')
def test_to_wkb_flavor_srid():
    with pytest.raises(ValueError, match='cannot be used together'):
        shapely.to_wkb(point_z, include_srid=True, flavor='iso')