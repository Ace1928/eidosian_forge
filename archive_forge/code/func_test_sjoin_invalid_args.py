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
@pytest.mark.parametrize('dfs', ['default-index', 'string-index'], indirect=True)
def test_sjoin_invalid_args(self, dfs):
    index, df1, df2, expected = dfs
    with pytest.raises(ValueError, match="'left_df' should be GeoDataFrame"):
        sjoin(df1.geometry, df2)
    with pytest.raises(ValueError, match="'right_df' should be GeoDataFrame"):
        sjoin(df1, df2.geometry)