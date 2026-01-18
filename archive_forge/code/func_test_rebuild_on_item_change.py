from math import sqrt
from shapely.geometry import (
from numpy.testing import assert_array_equal
import geopandas
from geopandas import _compat as compat
from geopandas import GeoDataFrame, GeoSeries, read_file, datasets
import pytest
import numpy as np
import pandas as pd
def test_rebuild_on_item_change(self):
    s = GeoSeries([Point(0, 0)])
    original_index = s.sindex
    s.iloc[0] = Point(0, 0)
    assert s.sindex is not original_index