from math import sqrt
from shapely.geometry import (
from numpy.testing import assert_array_equal
import geopandas
from geopandas import _compat as compat
from geopandas import GeoDataFrame, GeoSeries, read_file, datasets
import pytest
import numpy as np
import pandas as pd
def test_rebuild_on_slice(self):
    s = GeoSeries([Point(0, 0), Point(0, 0)])
    original_index = s.sindex
    sliced = s.iloc[:1]
    assert sliced.sindex is not original_index
    sliced = s.iloc[:]
    assert sliced.sindex is original_index
    sliced = s.iloc[::-1]
    assert sliced.sindex is not original_index