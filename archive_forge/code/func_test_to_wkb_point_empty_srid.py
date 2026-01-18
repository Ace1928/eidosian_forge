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
def test_to_wkb_point_empty_srid():
    expected = shapely.set_srid(empty_point, 4236)
    wkb = shapely.to_wkb(expected, include_srid=True)
    actual = shapely.from_wkb(wkb)
    assert shapely.get_srid(actual) == 4236