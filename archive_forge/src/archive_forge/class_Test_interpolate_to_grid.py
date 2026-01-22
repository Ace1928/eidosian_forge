import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import pytest
import cartopy.crs as ccrs
import cartopy.vector_transform as vec_trans
class Test_interpolate_to_grid:

    @classmethod
    def setup_class(cls):
        cls.x, cls.y = _sample_plate_carree_coordinates()
        cls.s = _sample_plate_carree_scalar_field()

    def test_data_extent(self):
        expected_x_grid = np.array([[-10.0, -5.0, 0.0, 5.0, 10.0], [-10.0, -5.0, 0.0, 5.0, 10.0], [-10.0, -5.0, 0.0, 5.0, 10.0]])
        expected_y_grid = np.array([[5.0, 5.0, 5.0, 5.0, 5.0], [7.5, 7.5, 7.5, 7.5, 7.5], [10.0, 10.0, 10.0, 10.0, 10]])
        expected_s_grid = np.array([[np.nan, 2.0, 3.0, 2.0, np.nan], [np.nan, 2.5, 3.5, 2.5, np.nan], [2.0, 3.0, 4.0, 3.0, 2.0]])
        x_grid, y_grid, s_grid = vec_trans._interpolate_to_grid(5, 3, self.x, self.y, self.s)
        assert_array_equal(x_grid, expected_x_grid)
        assert_array_equal(y_grid, expected_y_grid)
        assert_array_almost_equal(s_grid, expected_s_grid)

    def test_explicit_extent(self):
        expected_x_grid = np.array([[-5.0, 0.0, 5.0, 10.0], [-5.0, 0.0, 5.0, 10.0]])
        expected_y_grid = np.array([[7.5, 7.5, 7.5, 7.5], [10.0, 10.0, 10.0, 10]])
        expected_s_grid = np.array([[2.5, 3.5, 2.5, np.nan], [3.0, 4.0, 3.0, 2.0]])
        extent = (-5, 10, 7.5, 10)
        x_grid, y_grid, s_grid = vec_trans._interpolate_to_grid(4, 2, self.x, self.y, self.s, target_extent=extent)
        assert_array_equal(x_grid, expected_x_grid)
        assert_array_equal(y_grid, expected_y_grid)
        assert_array_almost_equal(s_grid, expected_s_grid)

    def test_multiple_fields(self):
        expected_x_grid = np.array([[-10.0, -5.0, 0.0, 5.0, 10.0], [-10.0, -5.0, 0.0, 5.0, 10.0], [-10.0, -5.0, 0.0, 5.0, 10.0]])
        expected_y_grid = np.array([[5.0, 5.0, 5.0, 5.0, 5.0], [7.5, 7.5, 7.5, 7.5, 7.5], [10.0, 10.0, 10.0, 10.0, 10]])
        expected_s_grid = np.array([[np.nan, 2.0, 3.0, 2.0, np.nan], [np.nan, 2.5, 3.5, 2.5, np.nan], [2.0, 3.0, 4.0, 3.0, 2.0]])
        x_grid, y_grid, s_grid1, s_grid2, s_grid3 = vec_trans._interpolate_to_grid(5, 3, self.x, self.y, self.s, self.s, self.s)
        assert_array_equal(x_grid, expected_x_grid)
        assert_array_equal(y_grid, expected_y_grid)
        assert_array_almost_equal(s_grid1, expected_s_grid)
        assert_array_almost_equal(s_grid2, expected_s_grid)
        assert_array_almost_equal(s_grid3, expected_s_grid)