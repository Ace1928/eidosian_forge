import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
def test_polygons_not_enough_points_in_shell_scalar():
    with pytest.raises(ValueError):
        shapely.polygons((1, 1))