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
def test_to_wkt_array_with_empty_z():
    empty_wkt = ['POINT Z EMPTY', None, 'POLYGON Z EMPTY']
    empty_geoms = shapely.from_wkt(empty_wkt)
    if shapely.geos_version < (3, 9, 0):
        empty_wkt = ['POINT EMPTY', None, 'POLYGON EMPTY']
    assert list(shapely.to_wkt(empty_geoms)) == empty_wkt