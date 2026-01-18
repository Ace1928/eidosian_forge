import re
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest
import cartopy.crs as ccrs
from cartopy.mpl import _MPL_38
from cartopy.tests.conftest import requires_scipy
@PARAMETRIZE_PCOLORMESH_WRAP
def test_pcolormesh_get_array_with_mask(mesh_data_kind):
    nx, ny = (36, 18)
    xbnds = np.linspace(0, 360, nx, endpoint=True)
    ybnds = np.linspace(-90, 90, ny, endpoint=True)
    x, y = np.meshgrid(xbnds, ybnds)
    data = np.exp(np.sin(np.deg2rad(x)) + np.cos(np.deg2rad(y)))
    data[5, :] = np.nan
    data = data[:-1, :-1]
    data = _to_rgb(data, mesh_data_kind)
    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1, projection=ccrs.PlateCarree())
    c = ax.pcolormesh(xbnds, ybnds, data, transform=ccrs.PlateCarree())
    assert c._wrapped_collection_fix is not None, 'No pcolormesh wrapping was done when it should have been.'
    result = c.get_array()
    np.testing.assert_array_equal(np.ma.getmask(result), np.isnan(data))
    np.testing.assert_array_equal(data, result, err_msg='Data supplied does not match data retrieved in wrapped case')
    ax.coastlines()
    ax.set_global()
    nx, ny = (36, 18)
    xbnds = np.linspace(-60, 60, nx, endpoint=True)
    ybnds = np.linspace(-80, 80, ny, endpoint=True)
    x, y = np.meshgrid(xbnds, ybnds)
    data = np.exp(np.sin(np.deg2rad(x)) + np.cos(np.deg2rad(y)))
    data[5, :] = np.nan
    data2 = data[:-1, :-1]
    data2 = _to_rgb(data2, mesh_data_kind)
    ax = fig.add_subplot(2, 1, 2, projection=ccrs.PlateCarree())
    c = ax.pcolormesh(xbnds, ybnds, data2, transform=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_global()
    assert getattr(c, '_wrapped_collection_fix', None) is None, 'pcolormesh wrapping was done when it should not have been.'
    result = c.get_array()
    expected = data2
    if not _MPL_38:
        expected = expected.ravel()
    np.testing.assert_array_equal(np.ma.getmask(result), np.isnan(expected))
    np.testing.assert_array_equal(expected, result, 'Data supplied does not match data retrieved in unwrapped case')