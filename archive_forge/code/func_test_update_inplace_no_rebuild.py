from math import sqrt
from shapely.geometry import (
from numpy.testing import assert_array_equal
import geopandas
from geopandas import _compat as compat
from geopandas import GeoDataFrame, GeoSeries, read_file, datasets
import pytest
import numpy as np
import pandas as pd
def test_update_inplace_no_rebuild(self):
    gdf = self.df.copy()
    old_sindex = gdf.sindex
    gdf.rename(columns={'A': 'AA'}, inplace=True)
    assert gdf.has_sindex
    new_sindex = gdf.sindex
    assert old_sindex is new_sindex