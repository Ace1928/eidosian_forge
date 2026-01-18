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
def test_sjoin_right(self):
    df = sjoin(self.pointdf, self.polydf, how='right')
    df2 = sjoin(self.polydf, self.pointdf, how='left')
    assert df.shape == (12, 8)
    assert df.shape == df2.shape
    for i, row in df.iterrows():
        assert row.geometry.geom_type == 'MultiPolygon'
    for i, row in df2.iterrows():
        assert row.geometry.geom_type == 'MultiPolygon'