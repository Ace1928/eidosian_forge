import numpy as np
from numpy.testing import assert_almost_equal
import pyproj
import pytest
import cartopy.crs as ccrs
from .helpers import check_proj_params
def test_offset(self):
    crs = ccrs.Sinusoidal()
    crs_offset = ccrs.Sinusoidal(false_easting=1234, false_northing=-4321)
    other_args = {'ellps=WGS84', 'lon_0=0.0', 'x_0=1234', 'y_0=-4321'}
    check_proj_params('sinu', crs_offset, other_args)
    assert tuple(np.array(crs.x_limits) + 1234) == crs_offset.x_limits
    assert tuple(np.array(crs.y_limits) - 4321) == crs_offset.y_limits