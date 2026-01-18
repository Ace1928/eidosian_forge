import copy
from io import BytesIO
import os
from pathlib import Path
import pickle
import warnings
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
from numpy.testing import assert_array_almost_equal as assert_arr_almost_eq
import pyproj
import pytest
import shapely.geometry as sgeom
import cartopy.crs as ccrs
@pytest.mark.parametrize('approx', [True, False])
def test_osni(self, approx):
    osni = ccrs.OSNI(approx=approx)
    ll = ccrs.Geodetic()
    lat, lon = np.array([54.5622169298669, -5.54159863617957], dtype=np.double)
    east, north = np.array([359000, 371000], dtype=np.double)
    assert_arr_almost_eq(osni.transform_point(lon, lat, ll), np.array([east, north]), -1)
    assert_arr_almost_eq(ll.transform_point(east, north, osni), np.array([lon, lat]), 3)