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
def test_transform_points_outside_domain():
    """Test CRS.transform_points with out of domain arrays."""
    crs = ccrs.Orthographic()
    result = crs.transform_points(ccrs.PlateCarree(), np.array([-120]), np.array([80]))
    assert np.all(np.isnan(result))
    result = crs.transform_points(ccrs.PlateCarree(), np.array([-120]), np.array([80]), trap=True)
    assert np.all(np.isnan(result))
    result = crs.transform_points(ccrs.PlateCarree(), np.array([-120, -120]), np.array([80, 80]))
    assert np.all(~np.isfinite(result[..., :2]))
    result = crs.transform_point(-120, 80, ccrs.PlateCarree())
    assert np.all(np.isnan(result))