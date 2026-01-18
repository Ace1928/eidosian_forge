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
@pytest.mark.skipif(shapely.geos_version < (3, 10, 0), reason='GEOS < 3.10')
@pytest.mark.parametrize('indent', [None, 0, 4])
def test_to_geojson_indent(indent):
    separators = (',', ':') if indent is None else (',', ': ')
    expected = json.dumps(json.loads(GEOJSON_GEOMETRY), indent=indent, separators=separators)
    actual = shapely.to_geojson(GEOJSON_GEOMETRY_EXPECTED, indent=indent)
    assert actual == expected