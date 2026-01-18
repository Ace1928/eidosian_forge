import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import pytest
import cartopy.crs as ccrs
import cartopy.vector_transform as vec_trans
def test_with_transform(self):
    target_crs = ccrs.PlateCarree()
    src_crs = ccrs.NorthPolarStereo()
    input_coords = [src_crs.transform_point(xp, yp, target_crs) for xp, yp in zip(self.x, self.y)]
    x_nps = np.array([ic[0] for ic in input_coords])
    y_nps = np.array([ic[1] for ic in input_coords])
    u_nps, v_nps = src_crs.transform_vectors(target_crs, self.x, self.y, self.u, self.v)
    expected_x_grid = np.array([[-10.0, -5.0, 0.0, 5.0, 10.0], [-10.0, -5.0, 0.0, 5.0, 10.0], [-10.0, -5.0, 0.0, 5.0, 10.0]])
    expected_y_grid = np.array([[5.0, 5.0, 5.0, 5.0, 5.0], [7.5, 7.5, 7.5, 7.5, 7.5], [10.0, 10.0, 10.0, 10.0, 10]])
    expected_u_grid = np.array([[np.nan, np.nan, np.nan, np.nan, np.nan], [np.nan, 2.3838, 3.5025, 2.6152, np.nan], [2, 3.0043, 4, 2.9022, 2]])
    expected_v_grid = np.array([[np.nan, np.nan, np.nan, np.nan, np.nan], [np.nan, 2.6527, 2.1904, 2.4192, np.nan], [5.5, 4.6483, 4, 4.47, 5.5]])
    x_grid, y_grid, u_grid, v_grid = vec_trans.vector_scalar_to_grid(src_crs, target_crs, (5, 3), x_nps, y_nps, u_nps, v_nps)
    assert_array_almost_equal(x_grid, expected_x_grid)
    assert_array_almost_equal(y_grid, expected_y_grid)
    assert_array_almost_equal(u_grid, expected_u_grid, decimal=4)
    assert_array_almost_equal(v_grid, expected_v_grid, decimal=4)