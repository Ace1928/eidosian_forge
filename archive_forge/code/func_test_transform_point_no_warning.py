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
def test_transform_point_no_warning():
    p = ccrs.PlateCarree()
    p2 = ccrs.Mercator()
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        p2.transform_point(1, 2, p)