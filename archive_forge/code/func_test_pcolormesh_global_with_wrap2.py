import re
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest
import cartopy.crs as ccrs
from cartopy.mpl import _MPL_38
from cartopy.tests.conftest import requires_scipy
@PARAMETRIZE_PCOLORMESH_WRAP
@pytest.mark.natural_earth
@pytest.mark.mpl_image_compare(filename='pcolormesh_global_wrap2.png', tolerance=1.87)
def test_pcolormesh_global_with_wrap2(mesh_data_kind):
    nx, ny = (36, 18)
    xbnds, xstep = np.linspace(0, 360, nx - 1, retstep=True, endpoint=True)
    ybnds, ystep = np.linspace(-90, 90, ny - 1, retstep=True, endpoint=True)
    xbnds -= xstep / 2
    ybnds -= ystep / 2
    xbnds = np.append(xbnds, xbnds[-1] + xstep)
    ybnds = np.append(ybnds, ybnds[-1] + ystep)
    x, y = np.meshgrid(xbnds, ybnds)
    data = np.exp(np.sin(np.deg2rad(x)) + np.cos(np.deg2rad(y)))
    data = data[:-1, :-1]
    fig = plt.figure()
    data = _to_rgb(data, mesh_data_kind)
    ax = fig.add_subplot(2, 1, 1, projection=ccrs.PlateCarree())
    ax.pcolormesh(xbnds, ybnds, data, transform=ccrs.PlateCarree(), snap=False)
    ax.coastlines()
    ax.set_global()
    ax = fig.add_subplot(2, 1, 2, projection=ccrs.PlateCarree(180))
    ax.pcolormesh(xbnds, ybnds, data, transform=ccrs.PlateCarree(), snap=False)
    ax.coastlines()
    ax.set_global()
    return fig