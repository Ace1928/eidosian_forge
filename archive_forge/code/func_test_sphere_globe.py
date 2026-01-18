import numpy as np
from numpy.testing import assert_almost_equal
import pytest
import cartopy.crs as ccrs
from .helpers import check_proj_params
def test_sphere_globe():
    globe = ccrs.Globe(semimajor_axis=1000, ellipse=None)
    ortho = ccrs.Orthographic(globe=globe)
    other_args = {'a=1000', 'lon_0=0.0', 'lat_0=0.0'}
    check_proj_params('ortho', ortho, other_args)
    assert_almost_equal(ortho.x_limits, [-999.99, 999.99])
    assert_almost_equal(ortho.y_limits, [-999.99, 999.99])