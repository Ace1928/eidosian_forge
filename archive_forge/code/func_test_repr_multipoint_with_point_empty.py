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
@pytest.mark.skipif(shapely.geos_version >= (3, 9, 0), reason='MULTIPOINT (EMPTY, 2 3) gives Exception on GEOS < 3.9')
def test_repr_multipoint_with_point_empty():
    geom = shapely.multipoints([point, empty_point])
    assert repr(geom) == '<shapely.MultiPoint Exception in WKT writer>'