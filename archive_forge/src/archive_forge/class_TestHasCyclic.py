import numpy as np
import numpy.ma as ma
from numpy.testing import assert_array_equal
import pytest
from cartopy.util import add_cyclic, add_cyclic_point, has_cyclic
class TestHasCyclic:
    """
    Test def has_cyclic(x, axis=-1, cyclic=360, precision=1e-4):
    - variations of x with and without axis keyword
    - different unit of x - cyclic keyword
    - detection of cyclic points - precision keyword
    """
    lons = np.arange(0, 360, 60)
    lats = np.arange(-90, 90, 180 / 5)
    lon2d, lat2d = np.meshgrid(lons, lats)
    lon3d = np.repeat(lon2d, 4).reshape((*lon2d.shape, 4))
    c_lons = np.concatenate((lons, np.array([360])))
    c_lon2d = np.concatenate((lon2d, np.full((lon2d.shape[0], 1), 360)), axis=1)
    c_lon3d = np.concatenate((lon3d, np.full((lon3d.shape[0], 1, lon3d.shape[2]), 360)), axis=1)

    @pytest.mark.parametrize('lon, clon', [(lons, c_lons), (lon2d, c_lon2d)])
    def test_data(self, lon, clon):
        """Test lon is not cyclic, clon is cyclic"""
        assert not has_cyclic(lon)
        assert has_cyclic(clon)

    @pytest.mark.parametrize('lon, clon, axis', [(lons, c_lons, 0), (lon2d, c_lon2d, 1), (ma.masked_inside(lon2d, 100, 200), ma.masked_inside(c_lon2d, 100, 200), 1)])
    def test_data_axis(self, lon, clon, axis):
        """Test lon is not cyclic, clon is cyclic, with axis keyword"""
        assert not has_cyclic(lon, axis=axis)
        assert has_cyclic(clon, axis=axis)

    def test_3d_axis(self):
        """Test 3d with axis keyword, no keyword name for axis"""
        assert has_cyclic(self.c_lon3d, 1)
        assert not has_cyclic(self.lon3d, 1)

    def test_3d_axis_cyclic(self):
        """Test 3d with axis and cyclic keywords"""
        new_clons = np.deg2rad(self.c_lon3d)
        new_lons = np.deg2rad(self.lon3d)
        assert has_cyclic(new_clons, axis=1, cyclic=np.deg2rad(360))
        assert not has_cyclic(new_lons, axis=1, cyclic=np.deg2rad(360))

    def test_1d_precision(self):
        """Test 1d with precision keyword"""
        new_clons = np.concatenate((self.lons, np.array([360 + 0.001])))
        assert has_cyclic(new_clons, precision=0.01)
        assert not has_cyclic(new_clons, precision=0.0002)