import warnings
import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest
import cartopy.crs as ccrs
def test_transform_and_inverse(self):
    x = np.arange(-60, 42.5, 2.5)
    y = np.arange(30, 72.5, 2.5)
    x2d, y2d = np.meshgrid(x, y)
    u = np.cos(np.deg2rad(y2d))
    v = np.cos(2.0 * np.deg2rad(x2d))
    src_proj = ccrs.PlateCarree()
    target_proj = ccrs.Stereographic(central_latitude=90, central_longitude=0)
    proj_xyz = target_proj.transform_points(src_proj, x2d, y2d)
    xt, yt = (proj_xyz[..., 0], proj_xyz[..., 1])
    ut, vt = target_proj.transform_vectors(src_proj, x2d, y2d, u, v)
    utt, vtt = src_proj.transform_vectors(target_proj, xt, yt, ut, vt)
    assert_array_almost_equal(u, utt, decimal=4)
    assert_array_almost_equal(v, vtt, decimal=4)