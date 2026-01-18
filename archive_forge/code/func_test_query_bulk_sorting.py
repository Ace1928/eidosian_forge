from math import sqrt
from shapely.geometry import (
from numpy.testing import assert_array_equal
import geopandas
from geopandas import _compat as compat
from geopandas import GeoDataFrame, GeoSeries, read_file, datasets
import pytest
import numpy as np
import pandas as pd
@pytest.mark.parametrize('sort, expected', ((True, [[0, 0, 0], [0, 1, 2]]), (False, [[0, 0, 0], [0, 1, 2]])))
def test_query_bulk_sorting(self, sort, expected):
    """Check that results from `query_bulk` don't depend
        on the order of geometries.
        """
    test_polys = GeoSeries([Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])])
    tree_polys = GeoSeries([Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]), Polygon([(-1, 1), (1, 1), (1, 3), (-1, 3)]), Polygon([(3, 3), (5, 3), (5, 5), (3, 5)])])
    tree_df = geopandas.GeoDataFrame(geometry=tree_polys)
    test_df = geopandas.GeoDataFrame(geometry=test_polys)
    res = tree_df.sindex.query(test_df.geometry, sort=sort)
    assert sorted(res[0]) == sorted(expected[0])
    assert sorted(res[1]) == sorted(expected[1])
    try:
        assert_array_equal(res, expected)
    except AssertionError as e:
        if sort is False:
            pytest.xfail('rtree results are known to be unordered, see https://github.com/geopandas/geopandas/issues/1337\nExpected:\n {}\n'.format(expected) + 'Got:\n {}\n'.format(res.tolist()))
        raise e