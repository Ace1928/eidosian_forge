from unittest import mock
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import pytest
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes, GeoAxesSubplot, InterProjectionTransform
@pytest.mark.natural_earth
def test_single_geometry(self):
    proj = ccrs.PlateCarree()
    ax = GeoAxes(plt.figure(), [0, 0, 1, 1], projection=proj)
    ax.add_geometries(next(cfeature.COASTLINE.geometries()), crs=proj)