from math import sqrt
from shapely.geometry import (
from numpy.testing import assert_array_equal
import geopandas
from geopandas import _compat as compat
from geopandas import GeoDataFrame, GeoSeries, read_file, datasets
import pytest
import numpy as np
import pandas as pd
def test_query_bulk_empty_input_array(self):
    """Tests the `query_bulk` method with an empty input array."""
    test_array = np.array([], dtype=object)
    expected_value = [[], []]
    res = self.df.sindex.query(test_array)
    assert_array_equal(res, expected_value)