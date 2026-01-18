import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
def test_points_from_xy():
    actual = shapely.points(2, [0, 1])
    assert_geometries_equal(actual, [shapely.Point(2, 0), shapely.Point(2, 1)])