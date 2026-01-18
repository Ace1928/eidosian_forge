import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
def test_offset_curve_distance_array():
    result = shapely.offset_curve([line_string, line_string], [-2.0, -3.0])
    assert result[0] == shapely.offset_curve(line_string, -2.0)
    assert result[1] == shapely.offset_curve(line_string, -3.0)