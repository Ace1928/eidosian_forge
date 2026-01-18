import re
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest
import cartopy.crs as ccrs
from cartopy.mpl import _MPL_38
from cartopy.tests.conftest import requires_scipy
@pytest.mark.natural_earth
@pytest.mark.mpl_image_compare(filename='pcolormesh_single_column_wrap.png')
def test_pcolormesh_single_column_wrap():
    ny = 36
    xbnds = np.array([360.9485619, 364.71999105])
    ybnds = np.linspace(-23.59000015, 24.81000137, ny, endpoint=True)
    x, y = np.meshgrid(xbnds, ybnds)
    data = np.sin(np.deg2rad(x)) / 10.0 + np.exp(np.cos(np.deg2rad(y)))
    data = data[:-1, :-1]
    rp = ccrs.RotatedPole(pole_longitude=177.5, pole_latitude=37.5)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(180))
    ax.pcolormesh(xbnds, ybnds, data, transform=rp, cmap='Spectral', snap=False)
    ax.coastlines()
    ax.set_global()
    return fig