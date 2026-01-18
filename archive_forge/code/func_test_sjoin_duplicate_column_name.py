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
def test_sjoin_duplicate_column_name(self):
    pointdf2 = self.pointdf.rename(columns={'pointattr1': 'Shape_Area'})
    df = sjoin(pointdf2, self.polydf, how='left')
    assert 'Shape_Area_left' in df.columns
    assert 'Shape_Area_right' in df.columns