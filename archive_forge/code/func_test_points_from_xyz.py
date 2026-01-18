import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
def test_points_from_xyz():
    actual = shapely.points(1, 1, [0, 1])
    assert_geometries_equal(actual, [shapely.Point(1, 1, 0), shapely.Point(1, 1, 1)])