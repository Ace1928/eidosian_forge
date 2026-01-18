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
def test_utm(self):
    utm30n = ccrs.UTM(30)
    ll = ccrs.Geodetic()
    lat, lon = np.array([51.5, -3.0], dtype=np.double)
    east, north = np.array([500000, 5705429.2], dtype=np.double)
    assert_arr_almost_eq(utm30n.transform_point(lon, lat, ll), [east, north], decimal=1)
    assert_arr_almost_eq(ll.transform_point(east, north, utm30n), [lon, lat], decimal=1)
    utm38s = ccrs.UTM(38, southern_hemisphere=True)
    lat, lon = np.array([-18.92, 47.5], dtype=np.double)
    east, north = np.array([763316.7, 7906160.8], dtype=np.double)
    assert_arr_almost_eq(utm38s.transform_point(lon, lat, ll), [east, north], decimal=1)
    assert_arr_almost_eq(ll.transform_point(east, north, utm38s), [lon, lat], decimal=1)