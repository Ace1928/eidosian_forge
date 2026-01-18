from math import sqrt
from shapely.geometry import (
from numpy.testing import assert_array_equal
import geopandas
from geopandas import _compat as compat
from geopandas import GeoDataFrame, GeoSeries, read_file, datasets
import pytest
import numpy as np
import pandas as pd
@pytest.mark.skipif(compat.USE_PYGEOS or compat.USE_SHAPELY_20, reason='RTree supports sindex.nearest with different behaviour')
def test_rtree_nearest_warns(self):
    df = geopandas.GeoDataFrame({'geometry': []})
    with pytest.warns(FutureWarning, match='sindex.nearest using the rtree backend'):
        df.sindex.nearest((0, 0, 1, 1), num_results=2)