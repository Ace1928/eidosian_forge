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
@pytest.mark.skipif(not (compat.USE_PYGEOS or compat.USE_SHAPELY_20), reason='array input in interpolate is not implemented for shapely<2')
@pytest.mark.parametrize('size', [10, 20, 50])
def test_sample_points(self, size):
    for gs in (self.g1, self.na, self.a1, self.na_none):
        output = gs.sample_points(size)
        assert_index_equal(gs.index, output.index)
        assert len(output.explode(ignore_index=True)) == len(gs[~(gs.is_empty | gs.isna())]) * size
    with pytest.warns(FutureWarning, match="The 'seed' keyword is deprecated"):
        _ = gs.sample_points(size, seed=1)