import io
from unittest import mock
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
def test_pcolormesh_partially_masked():
    data = np.ma.masked_all((39, 29))
    data[0:100] = 10
    with mock.patch('cartopy.mpl.geoaxes.GeoAxes.pcolor') as pcolor:
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.pcolormesh(np.linspace(0, 360, 30), np.linspace(-90, 90, 40), data)
        assert pcolor.call_count == 1, 'pcolor should have been called exactly once.'