import re
import numpy as np
import pytest
from mpl_toolkits.axisartist.angle_helper import (
@pytest.mark.parametrize('args, kwargs, expected_levels, expected_factor', [((-180, 180, 10), {'hour': False}, np.arange(-180, 181, 30), 1.0), ((-12, 12, 10), {'hour': True}, np.arange(-12, 13, 2), 1.0)])
def test_select_step(args, kwargs, expected_levels, expected_factor):
    levels, n, factor = select_step(*args, **kwargs)
    assert n == len(levels)
    np.testing.assert_array_equal(levels, expected_levels)
    assert factor == expected_factor