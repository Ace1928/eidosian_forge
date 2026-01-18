import json
import os
import random
import re
import shutil
import tempfile
import warnings
import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
from pandas.testing import assert_index_equal
from pyproj import CRS
from shapely.geometry import (
from shapely.geometry.base import BaseGeometry
from geopandas import GeoSeries, GeoDataFrame, read_file, datasets, clip
from geopandas._compat import ignore_shapely2_warnings
from geopandas.array import GeometryArray, GeometryDtype
from geopandas.testing import assert_geoseries_equal, geom_almost_equals
from geopandas.tests.util import geom_equals
from pandas.testing import assert_series_equal
import pytest
def test_proj4strings(self):
    reprojected = self.g3.to_crs('+proj=utm +zone=30')
    reprojected_back = reprojected.to_crs(epsg=4326)
    assert geom_almost_equals(self.g3, reprojected_back)
    reprojected = self.g3.to_crs({'proj': 'utm', 'zone': '30'})
    reprojected_back = reprojected.to_crs(epsg=4326)
    assert geom_almost_equals(self.g3, reprojected_back)
    copy = self.g3.copy()
    copy.crs = 'epsg:4326'
    reprojected = copy.to_crs({'proj': 'utm', 'zone': '30'})
    reprojected_back = reprojected.to_crs(epsg=4326)
    assert geom_almost_equals(self.g3, reprojected_back)
    reprojected_string = self.g3.to_crs('+proj=utm +zone=30')
    reprojected_dict = self.g3.to_crs({'proj': 'utm', 'zone': '30'})
    assert geom_almost_equals(reprojected_string, reprojected_dict)