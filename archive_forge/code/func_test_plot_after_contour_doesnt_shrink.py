import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest
import cartopy.crs as ccrs
from cartopy.tests.conftest import requires_scipy
@pytest.mark.parametrize('func', ['contour', 'contourf'])
def test_plot_after_contour_doesnt_shrink(func):
    xglobal = np.linspace(-180, 180)
    yglobal = np.linspace(-90, 90.00001)
    data = np.hypot(*np.meshgrid(xglobal, yglobal))
    target_proj = ccrs.PlateCarree(central_longitude=200)
    source_proj = ccrs.PlateCarree()
    ax = plt.axes(projection=target_proj)
    test_func = getattr(ax, func)
    test_func(xglobal, yglobal, data, transform=source_proj)
    ax.plot([10, 20], [20, 30], transform=source_proj)
    expected = np.array([xglobal[0], xglobal[-1], yglobal[0], 90])
    assert_array_almost_equal(ax.get_extent(), expected)