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
@pytest.mark.mpl_image_compare(filename='pcolormesh_mercator_wrap.png')
def test_pcolormesh_wrap_set_array(mesh_data_kind):
    x = np.linspace(0, 360, 73)
    y = np.linspace(-87.5, 87.5, 36)
    X, Y = np.meshgrid(*[np.deg2rad(c) for c in (x, y)])
    Z = np.cos(Y) + 0.375 * np.sin(2.0 * X)
    Z = Z[:-1, :-1]
    Z = _to_rgb(Z, mesh_data_kind)
    ax = plt.axes(projection=ccrs.Mercator())
    norm = plt.Normalize(np.min(Z), np.max(Z))
    ax.coastlines()
    coll = ax.pcolormesh(x, y, np.ones(Z.shape), norm=norm, transform=ccrs.PlateCarree(), snap=False)
    coll.set_array(Z)
    return ax.figure