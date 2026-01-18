import math
from typing import Sequence
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Point, Polygon, GeometryCollection
import geopandas
import geopandas._compat as compat
from geopandas import GeoDataFrame, GeoSeries, read_file, sjoin, sjoin_nearest
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from pandas.testing import assert_frame_equal, assert_series_equal
import pytest
@pytest.mark.parametrize('how, lsuffix, rsuffix, expected_cols', [('left', 'left', 'right', {'col_left', 'col_right', 'index_right'}), ('inner', 'left', 'right', {'col_left', 'col_right', 'index_right'}), ('right', 'left', 'right', {'col_left', 'col_right', 'index_left'}), ('left', 'lft', 'rgt', {'col_lft', 'col_rgt', 'index_rgt'}), ('inner', 'lft', 'rgt', {'col_lft', 'col_rgt', 'index_rgt'}), ('right', 'lft', 'rgt', {'col_lft', 'col_rgt', 'index_lft'})])
def test_suffixes(self, how: str, lsuffix: str, rsuffix: str, expected_cols):
    left = GeoDataFrame({'col': [1], 'geometry': [Point(0, 0)]})
    right = GeoDataFrame({'col': [1], 'geometry': [Point(0, 0)]})
    joined = sjoin(left, right, how=how, lsuffix=lsuffix, rsuffix=rsuffix)
    assert set(joined.columns) == expected_cols | {'geometry'}