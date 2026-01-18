import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import cartopy.crs as ccrs
def test_update_lim():
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.update_datalim([(-10, -10), (-5, -5)])
    assert_array_almost_equal(ax.dataLim.get_points(), np.array([[-10.0, -10.0], [-5.0, -5.0]]))