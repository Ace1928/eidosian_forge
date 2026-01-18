import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
def test_polygonize_missing():
    result = shapely.polygonize([None, None])
    assert result == GeometryCollection()