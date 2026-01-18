import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import cartopy.crs as ccrs
def test_view_lim_autoscaling():
    x = np.linspace(0.12910209, 0.42141822)
    y = np.linspace(0.03739792, 0.33029076)
    x, y = np.meshgrid(x, y)
    ax = plt.axes(projection=ccrs.RotatedPole(37.5, 357.5))
    ax.scatter(x, y, x * y, transform=ccrs.PlateCarree())
    expected = np.array([[86.12433701, 52.51570463], [86.69696603, 52.86372057]])
    assert_array_almost_equal(ax.viewLim.frozen().get_points(), expected, decimal=2)
    plt.draw()
    assert_array_almost_equal(ax.viewLim.frozen().get_points(), expected, decimal=2)
    ax.autoscale_view(tight=False)
    expected_non_tight = np.array([[86, 52.45], [86.8, 52.9]])
    assert_array_almost_equal(ax.viewLim.frozen().get_points(), expected_non_tight, decimal=1)