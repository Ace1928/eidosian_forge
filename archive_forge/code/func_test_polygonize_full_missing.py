import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
def test_polygonize_full_missing():
    result = shapely.polygonize_full([None, None])
    assert len(result) == 4
    assert all((geom == GeometryCollection() for geom in result))