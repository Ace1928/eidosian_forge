from math import sqrt
from shapely.geometry import (
from numpy.testing import assert_array_equal
import geopandas
from geopandas import _compat as compat
from geopandas import GeoDataFrame, GeoSeries, read_file, datasets
import pytest
import numpy as np
import pandas as pd
@pytest.mark.parametrize('test_geoms, expected_value', [([GeometryCollection()], [[], []]), ([GeometryCollection(), None], [[], []]), ([None], [[], []]), ([None, box(-0.5, -0.5, 0.5, 0.5), None], [[1], [0]])])
def test_query_bulk_empty_geometry(self, test_geoms, expected_value):
    """Tests the `query_bulk` method with an empty geometry."""
    test_geoms = geopandas.GeoSeries(test_geoms, index=range(len(test_geoms)))
    res = self.df.sindex.query(test_geoms)
    assert_array_equal(res, expected_value)