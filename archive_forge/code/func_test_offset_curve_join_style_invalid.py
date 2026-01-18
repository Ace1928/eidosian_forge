import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
def test_offset_curve_join_style_invalid():
    with pytest.raises(ValueError, match="'invalid' is not a valid option"):
        shapely.offset_curve(line_string, 1.0, join_style='invalid')