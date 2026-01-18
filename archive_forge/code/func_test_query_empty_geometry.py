from math import sqrt
from shapely.geometry import (
from numpy.testing import assert_array_equal
import geopandas
from geopandas import _compat as compat
from geopandas import GeoDataFrame, GeoSeries, read_file, datasets
import pytest
import numpy as np
import pandas as pd
@pytest.mark.parametrize('test_geom, expected_value', [(None, []), (GeometryCollection(), []), (Point(), []), (MultiPolygon(), []), (Polygon(), [])])
def test_query_empty_geometry(self, test_geom, expected_value):
    """Tests the `query` method with empty geometry."""
    res = self.df.sindex.query(test_geom)
    assert_array_equal(res, expected_value)