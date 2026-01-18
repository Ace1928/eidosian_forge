from math import sqrt
from shapely.geometry import (
from numpy.testing import assert_array_equal
import geopandas
from geopandas import _compat as compat
from geopandas import GeoDataFrame, GeoSeries, read_file, datasets
import pytest
import numpy as np
import pandas as pd
def test_sindex_rebuild_on_set_geometry(self):
    assert self.df.sindex is not None
    original_index = self.df.sindex
    self.df.set_geometry([Point(x, y) for x, y in zip(range(5, 10), range(5, 10))], inplace=True)
    assert self.df.sindex is not original_index