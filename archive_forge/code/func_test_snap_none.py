import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
def test_snap_none():
    actual = shapely.snap(None, point, tolerance=1.0)
    assert actual is None