from math import sqrt
from shapely.geometry import (
from numpy.testing import assert_array_equal
import geopandas
from geopandas import _compat as compat
from geopandas import GeoDataFrame, GeoSeries, read_file, datasets
import pytest
import numpy as np
import pandas as pd
def test_rebuild_on_update_inplace(self):
    gdf = self.df.copy()
    old_sindex = gdf.sindex
    gdf.sort_values('A', ascending=False, inplace=True)
    assert not gdf.has_sindex
    new_sindex = gdf.sindex
    assert new_sindex is not old_sindex
    assert gdf.index.tolist() == [4, 3, 2, 1, 0]