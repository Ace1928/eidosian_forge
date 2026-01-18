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
def test_to_wkt():
    point = shapely.points(1, 1)
    actual = shapely.to_wkt(point)
    assert actual == 'POINT (1 1)'
    actual = shapely.to_wkt(point, trim=False)
    assert actual == 'POINT (1.000000 1.000000)'
    actual = shapely.to_wkt(point, rounding_precision=3, trim=False)
    assert actual == 'POINT (1.000 1.000)'