from numpy.testing import assert_almost_equal
import pytest
import cartopy.crs as ccrs
from .helpers import check_proj_params
def test_scale_factor():
    scale_factor = 0.939692620786
    crs = ccrs.Mercator(scale_factor=scale_factor, globe=ccrs.Globe(ellipse='sphere'))
    other_args = {'ellps=sphere', 'lon_0=0.0', 'x_0=0.0', 'y_0=0.0', 'units=m', f'k_0={scale_factor:.12f}'}
    check_proj_params('merc', crs, other_args)
    assert_almost_equal(crs.boundary.bounds, [-18808021, -14585266, 18808021, 17653216], decimal=0)