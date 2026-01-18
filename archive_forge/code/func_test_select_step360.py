import re
import numpy as np
import pytest
from mpl_toolkits.axisartist.angle_helper import (
@pytest.mark.parametrize('args, kwargs, expected_levels, expected_factor', [((dms2float(20, 21.2), dms2float(21, 33.3), 5), {}, np.arange(1215, 1306, 15), 60.0), ((dms2float(20.5, seconds=21.2), dms2float(20.5, seconds=33.3), 5), {}, np.arange(73820, 73835, 2), 3600.0), ((dms2float(20, 21.2), dms2float(20, 53.3), 5), {}, np.arange(1220, 1256, 5), 60.0), ((21.2, 33.3, 5), {}, np.arange(20, 35, 2), 1.0), ((dms2float(20, 21.2), dms2float(21, 33.3), 5), {}, np.arange(1215, 1306, 15), 60.0), ((dms2float(20.5, seconds=21.2), dms2float(20.5, seconds=33.3), 5), {}, np.arange(73820, 73835, 2), 3600.0), ((dms2float(20.5, seconds=21.2), dms2float(20.5, seconds=21.4), 5), {}, np.arange(7382120, 7382141, 5), 360000.0), ((dms2float(20.5, seconds=11.2), dms2float(20.5, seconds=53.3), 5), {'threshold_factor': 60}, np.arange(12301, 12310), 600.0), ((dms2float(20.5, seconds=11.2), dms2float(20.5, seconds=53.3), 5), {'threshold_factor': 1}, np.arange(20502, 20517, 2), 1000.0)])
def test_select_step360(args, kwargs, expected_levels, expected_factor):
    levels, n, factor = select_step360(*args, **kwargs)
    assert n == len(levels)
    np.testing.assert_array_equal(levels, expected_levels)
    assert factor == expected_factor