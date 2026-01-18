import re
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest
import cartopy.crs as ccrs
from cartopy.mpl import _MPL_38
from cartopy.tests.conftest import requires_scipy
@pytest.mark.natural_earth
@pytest.mark.mpl_image_compare(filename='pcolormesh_mercator_wrap.png')
def test_pcolormesh_mercator_wrap():
    x = np.linspace(0, 360, 73)
    y = np.linspace(-87.5, 87.5, 36)
    X, Y = np.meshgrid(*[np.deg2rad(c) for c in (x, y)])
    Z = np.cos(Y) + 0.375 * np.sin(2.0 * X)
    Z = Z[:-1, :-1]
    ax = plt.axes(projection=ccrs.Mercator())
    ax.coastlines()
    ax.pcolormesh(x, y, Z, transform=ccrs.PlateCarree(), snap=False)
    return ax.figure