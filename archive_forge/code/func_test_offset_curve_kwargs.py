import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
def test_offset_curve_kwargs():
    result1 = shapely.offset_curve(line_string, -2.0, quad_segs=2, join_style='mitre', mitre_limit=2.0)
    result2 = shapely.offset_curve(line_string, -2.0)
    assert result1 != result2