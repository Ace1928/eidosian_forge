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
def _test_binary_operator(self, op, expected, a, b):
    """
        The operators only have GeoSeries on the left, but can have
        GeoSeries or GeoDataFrame on the right.
        If GeoDataFrame is on the left, geometry column is used.

        """
    if isinstance(expected, GeoPandasBase):
        fcmp = assert_geoseries_equal
    else:

        def fcmp(a, b):
            assert geom_equals(a, b)
    if isinstance(b, GeoPandasBase):
        right_df = True
    else:
        right_df = False
    self._binary_op_test(op, expected, a, b, fcmp, False, right_df)