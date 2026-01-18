import string
import warnings
import numpy as np
from numpy.testing import assert_array_equal
from pandas import DataFrame, Index, MultiIndex, Series, concat
import shapely
from shapely.geometry import (
from shapely.geometry.collection import GeometryCollection
from shapely.ops import unary_union
from shapely import wkt
from geopandas import GeoDataFrame, GeoSeries
from geopandas.base import GeoPandasBase
from geopandas.testing import assert_geodataframe_equal, geom_almost_equals
from geopandas.tests.util import assert_geoseries_equal, geom_equals
from geopandas import _compat as compat
from pandas.testing import assert_frame_equal, assert_index_equal, assert_series_equal
import pytest
def test_xyz_points(self):
    expected_x = [-73.9847, -74.0446]
    expected_y = [40.7484, 40.6893]
    expected_z = [30.3244, 31.2344]
    assert_array_dtype_equal(expected_x, self.landmarks.geometry.x)
    assert_array_dtype_equal(expected_y, self.landmarks.geometry.y)
    assert_array_dtype_equal(expected_z, self.landmarks.geometry.z)
    expected_z = [30.3244, 31.2344, np.nan]
    assert_array_dtype_equal(expected_z, self.landmarks_mixed.geometry.z)