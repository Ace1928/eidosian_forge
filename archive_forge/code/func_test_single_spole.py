from numpy.testing import assert_array_almost_equal
import pyproj
import pytest
import cartopy.crs as ccrs
from .helpers import check_proj_params
def test_single_spole(self):
    s_pole_crs = ccrs.LambertConformal(standard_parallels=[-1.0])
    expected_x = (-19939660, 19939660)
    expected_y = (-735590302, -8183795)
    if pyproj.__proj_version__ >= '9.2.0':
        expected_x = (-19840440, 19840440)
        expected_y = (-370239953, -8191953)
    print(s_pole_crs.x_limits)
    assert_array_almost_equal(s_pole_crs.x_limits, expected_x, decimal=0)
    assert_array_almost_equal(s_pole_crs.y_limits, expected_y, decimal=0)