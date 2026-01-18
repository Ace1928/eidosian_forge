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
def test_to_wkb_exceptions():
    with pytest.raises(TypeError):
        shapely.to_wkb(1)
    with pytest.raises(shapely.GEOSException):
        shapely.to_wkb(point, output_dimension=5)
    with pytest.raises(ValueError):
        shapely.to_wkb(point, flavor='other')