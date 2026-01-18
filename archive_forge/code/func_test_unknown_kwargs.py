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
@pytest.mark.parametrize('dfs', ['default-index'], indirect=True)
def test_unknown_kwargs(self, dfs):
    _, df1, df2, _ = dfs
    with pytest.raises(TypeError, match="sjoin\\(\\) got an unexpected keyword argument 'extra_param'"):
        sjoin(df1, df2, extra_param='test')