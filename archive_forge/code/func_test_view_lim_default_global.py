import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import cartopy.crs as ccrs
def test_view_lim_default_global(tmp_path):
    ax = plt.axes(projection=ccrs.PlateCarree())
    assert_array_almost_equal(ax.viewLim.frozen().get_points(), [[0, 0], [1, 1]])
    plt.savefig(tmp_path / 'view_lim_default_global.png')
    expected = np.array([[-180, -90], [180, 90]])
    assert_array_almost_equal(ax.viewLim.frozen().get_points(), expected)