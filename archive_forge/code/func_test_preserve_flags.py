import os
from packaging.version import Version
import warnings
import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
import shapely
from shapely.geometry import Point, GeometryCollection, LineString, LinearRing
import geopandas
from geopandas import GeoDataFrame, GeoSeries
import geopandas._compat as compat
from geopandas.array import from_shapely
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from pandas.testing import assert_frame_equal, assert_series_equal
import pytest
def test_preserve_flags(df):
    df = df.set_flags(allows_duplicate_labels=False)
    assert df.flags.allows_duplicate_labels is False
    for subset in [df[:2], df[df['value1'] > 2], df[['value2', 'geometry']]]:
        assert df.flags.allows_duplicate_labels is False
    df2 = df.reset_index()
    assert df2.flags.allows_duplicate_labels is False
    with pytest.raises(ValueError):
        df.reindex([0, 0, 1])
    with pytest.raises(ValueError):
        df[['value1', 'value1', 'geometry']]
    with pytest.raises(ValueError):
        pd.concat([df, df])