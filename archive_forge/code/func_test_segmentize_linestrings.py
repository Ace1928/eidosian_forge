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
@pytest.mark.skipif(shapely.geos.geos_version < (3, 10, 0), reason='requires GEOS>=3.10')
@pytest.mark.skipif(not (compat.USE_PYGEOS or compat.USE_SHAPELY_20), reason='segmentize keyword introduced in shapely 2.0')
def test_segmentize_linestrings(self):
    expected_g1 = GeoSeries([Polygon(((0, 0), (0.5, 0), (1, 0), (1, 0.5), (1, 1), (0.6666666666666666, 0.6666666666666666), (0.3333333333333333, 0.3333333333333333), (0, 0))), Polygon(((0, 0), (0.5, 0), (1, 0), (1, 0.5), (1, 1), (0.5, 1), (0, 1), (0, 0.5), (0, 0)))])
    expected_g5 = GeoSeries([LineString([(0, 0), (0, 0.5), (0, 1), (0.5, 1), (1, 1)]), LineString([(0, 0), (0.5, 0), (1, 0), (1, 0.5), (1, 1), (0.5, 1), (0, 1)])])
    result_g1 = self.g1.segmentize(max_segment_length=0.5)
    result_g5 = self.g5.segmentize(max_segment_length=0.5)
    assert_geoseries_equal(expected_g1, result_g1)
    assert_geoseries_equal(expected_g5, result_g5)