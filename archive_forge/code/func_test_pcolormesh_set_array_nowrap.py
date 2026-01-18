import re
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest
import cartopy.crs as ccrs
from cartopy.mpl import _MPL_38
from cartopy.tests.conftest import requires_scipy
def test_pcolormesh_set_array_nowrap():
    nx, ny = (36, 18)
    xbnds = np.linspace(-60, 60, nx, endpoint=True)
    ybnds = np.linspace(-80, 80, ny, endpoint=True)
    xbnds, ybnds = np.meshgrid(xbnds, ybnds)
    rng = np.random.default_rng()
    data = rng.random((ny - 1, nx - 1))
    ax = plt.figure().add_subplot(projection=ccrs.PlateCarree())
    mesh = ax.pcolormesh(xbnds, ybnds, data)
    assert not hasattr(mesh, '_wrapped_collection_fix')
    expected = data
    if not _MPL_38:
        expected = expected.ravel()
    np.testing.assert_array_equal(mesh.get_array(), expected)
    data = rng.random((nx - 1) * (ny - 1))
    mesh.set_array(data)
    np.testing.assert_array_equal(mesh.get_array(), data.reshape(ny - 1, nx - 1))
    data = rng.random((ny - 1, nx - 1))
    mesh.set_array(data)
    np.testing.assert_array_equal(mesh.get_array(), data)