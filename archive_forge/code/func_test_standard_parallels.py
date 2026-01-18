import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal
import pytest
import cartopy.crs as ccrs
from .helpers import check_proj_params
def test_standard_parallels(self):
    aea = ccrs.AlbersEqualArea(standard_parallels=(13, 37))
    other_args = {'ellps=WGS84', 'lon_0=0.0', 'lat_0=0.0', 'x_0=0.0', 'y_0=0.0', 'lat_1=13', 'lat_2=37'}
    check_proj_params('aea', aea, other_args)
    aea = ccrs.AlbersEqualArea(standard_parallels=(13,))
    other_args = {'ellps=WGS84', 'lon_0=0.0', 'lat_0=0.0', 'x_0=0.0', 'y_0=0.0', 'lat_1=13'}
    check_proj_params('aea', aea, other_args)
    aea = ccrs.AlbersEqualArea(standard_parallels=13)
    other_args = {'ellps=WGS84', 'lon_0=0.0', 'lat_0=0.0', 'x_0=0.0', 'y_0=0.0', 'lat_1=13'}
    check_proj_params('aea', aea, other_args)