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

        This is a helper to call a function on GeoSeries and GeoDataFrame
        arguments. For example, 'intersection' is a member of both GeoSeries
        and GeoDataFrame and can take either GeoSeries or GeoDataFrame inputs.
        This function has the ability to test all four combinations of input
        types.

        Parameters
        ----------

        expected : str
            The operation to be tested. e.g., 'intersection'
        left: GeoSeries
        right: GeoSeries
        fcmp: function
            Called with the result of the operation and expected. It should
            assert if the result is incorrect
        left_df: bool
            If the left input should also be called with a GeoDataFrame
        right_df: bool
            Indicates whether the right input should be called with a
            GeoDataFrame

        