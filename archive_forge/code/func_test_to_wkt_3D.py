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
def test_to_wkt_3D():
    point_z = shapely.points(1, 1, 1)
    actual = shapely.to_wkt(point_z)
    assert actual == 'POINT Z (1 1 1)'
    actual = shapely.to_wkt(point_z, output_dimension=3)
    assert actual == 'POINT Z (1 1 1)'
    actual = shapely.to_wkt(point_z, output_dimension=2)
    assert actual == 'POINT (1 1)'
    actual = shapely.to_wkt(point_z, old_3d=True)
    assert actual == 'POINT (1 1 1)'