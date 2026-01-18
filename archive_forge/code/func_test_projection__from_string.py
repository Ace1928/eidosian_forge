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
def test_projection__from_string():
    crs = ccrs.Projection('NAD83 / Pennsylvania South')
    assert crs.as_geocentric().datum.name == 'North American Datum 1983'
    assert_almost_equal(crs.bounds, [361633.1351868, 859794.6690229, 45575.5693199, 209415.9845754])