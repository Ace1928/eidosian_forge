import re
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest
import cartopy.crs as ccrs
from cartopy.mpl import _MPL_38
from cartopy.tests.conftest import requires_scipy
@pytest.mark.natural_earth
@pytest.mark.mpl_image_compare(filename='pcolormesh_global_wrap3.png', tolerance=1.42)
def test_pcolormesh_set_clim_with_mask():
    """Testing that set_clim works with masked arrays properly."""
    nx, ny = (33, 17)
    xbnds = np.linspace(-1.875, 358.125, nx, endpoint=True)
    ybnds = np.linspace(91.25, -91.25, ny, endpoint=True)
    xbnds, ybnds = np.meshgrid(xbnds, ybnds)
    data = np.exp(np.sin(np.deg2rad(xbnds)) + np.cos(np.deg2rad(ybnds)))
    ybnds = np.append(ybnds, ybnds[:, 1:2], axis=1)
    xbnds = np.append(xbnds, xbnds[:, 1:2] + 360, axis=1)
    data = np.ma.concatenate([data, data[:, 0:1]], axis=1)
    data = data[:-1, :-1]
    data = np.ma.masked_greater(data, 2.6)
    bad_initial_norm = plt.Normalize(-100, 100)
    fig = plt.figure()
    ax = fig.add_subplot(3, 1, 1, projection=ccrs.PlateCarree(-45))
    c = ax.pcolormesh(xbnds, ybnds, data, transform=ccrs.PlateCarree(), norm=bad_initial_norm, snap=False)
    assert c._wrapped_collection_fix is not None, 'No pcolormesh wrapping was done when it should have been.'
    ax.coastlines()
    ax.set_global()
    ax = fig.add_subplot(3, 1, 2, projection=ccrs.PlateCarree(-1.87499952))
    ax.pcolormesh(xbnds, ybnds, data, transform=ccrs.PlateCarree(), snap=False)
    ax.coastlines()
    ax.set_global()
    ax = fig.add_subplot(3, 1, 3, projection=ccrs.Robinson(-2))
    ax.pcolormesh(xbnds, ybnds, data, transform=ccrs.PlateCarree(), snap=False)
    ax.coastlines()
    ax.set_global()
    c.set_clim(data.min(), data.max())
    return fig