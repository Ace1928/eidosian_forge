from __future__ import annotations
from datetime import datetime
from itertools import product
import numpy as np
import pytest
from xarray import (
from xarray.core import dtypes
from xarray.core.combine import (
from xarray.tests import assert_equal, assert_identical, requires_cftime
from xarray.tests.test_dataset import create_test_data
class TestTileIDsFromCoords:

    def test_1d(self):
        ds0 = Dataset({'x': [0, 1]})
        ds1 = Dataset({'x': [2, 3]})
        expected = {(0,): ds0, (1,): ds1}
        actual, concat_dims = _infer_concat_order_from_coords([ds1, ds0])
        assert_combined_tile_ids_equal(expected, actual)
        assert concat_dims == ['x']

    def test_2d(self):
        ds0 = Dataset({'x': [0, 1], 'y': [10, 20, 30]})
        ds1 = Dataset({'x': [2, 3], 'y': [10, 20, 30]})
        ds2 = Dataset({'x': [0, 1], 'y': [40, 50, 60]})
        ds3 = Dataset({'x': [2, 3], 'y': [40, 50, 60]})
        ds4 = Dataset({'x': [0, 1], 'y': [70, 80, 90]})
        ds5 = Dataset({'x': [2, 3], 'y': [70, 80, 90]})
        expected = {(0, 0): ds0, (1, 0): ds1, (0, 1): ds2, (1, 1): ds3, (0, 2): ds4, (1, 2): ds5}
        actual, concat_dims = _infer_concat_order_from_coords([ds1, ds0, ds3, ds5, ds2, ds4])
        assert_combined_tile_ids_equal(expected, actual)
        assert concat_dims == ['x', 'y']

    def test_no_dimension_coords(self):
        ds0 = Dataset({'foo': ('x', [0, 1])})
        ds1 = Dataset({'foo': ('x', [2, 3])})
        with pytest.raises(ValueError, match='Could not find any dimension'):
            _infer_concat_order_from_coords([ds1, ds0])

    def test_coord_not_monotonic(self):
        ds0 = Dataset({'x': [0, 1]})
        ds1 = Dataset({'x': [3, 2]})
        with pytest.raises(ValueError, match='Coordinate variable x is neither monotonically increasing nor'):
            _infer_concat_order_from_coords([ds1, ds0])

    def test_coord_monotonically_decreasing(self):
        ds0 = Dataset({'x': [3, 2]})
        ds1 = Dataset({'x': [1, 0]})
        expected = {(0,): ds0, (1,): ds1}
        actual, concat_dims = _infer_concat_order_from_coords([ds1, ds0])
        assert_combined_tile_ids_equal(expected, actual)
        assert concat_dims == ['x']

    def test_no_concatenation_needed(self):
        ds = Dataset({'foo': ('x', [0, 1])})
        expected = {(): ds}
        actual, concat_dims = _infer_concat_order_from_coords([ds])
        assert_combined_tile_ids_equal(expected, actual)
        assert concat_dims == []

    def test_2d_plus_bystander_dim(self):
        ds0 = Dataset({'x': [0, 1], 'y': [10, 20, 30], 't': [0.1, 0.2]})
        ds1 = Dataset({'x': [2, 3], 'y': [10, 20, 30], 't': [0.1, 0.2]})
        ds2 = Dataset({'x': [0, 1], 'y': [40, 50, 60], 't': [0.1, 0.2]})
        ds3 = Dataset({'x': [2, 3], 'y': [40, 50, 60], 't': [0.1, 0.2]})
        expected = {(0, 0): ds0, (1, 0): ds1, (0, 1): ds2, (1, 1): ds3}
        actual, concat_dims = _infer_concat_order_from_coords([ds1, ds0, ds3, ds2])
        assert_combined_tile_ids_equal(expected, actual)
        assert concat_dims == ['x', 'y']

    def test_string_coords(self):
        ds0 = Dataset({'person': ['Alice', 'Bob']})
        ds1 = Dataset({'person': ['Caroline', 'Daniel']})
        expected = {(0,): ds0, (1,): ds1}
        actual, concat_dims = _infer_concat_order_from_coords([ds1, ds0])
        assert_combined_tile_ids_equal(expected, actual)
        assert concat_dims == ['person']

    def test_lexicographic_sort_string_coords(self):
        ds0 = Dataset({'simulation': ['run8', 'run9']})
        ds1 = Dataset({'simulation': ['run10', 'run11']})
        expected = {(0,): ds1, (1,): ds0}
        actual, concat_dims = _infer_concat_order_from_coords([ds1, ds0])
        assert_combined_tile_ids_equal(expected, actual)
        assert concat_dims == ['simulation']

    def test_datetime_coords(self):
        ds0 = Dataset({'time': [datetime(2000, 3, 6), datetime(2001, 3, 7)]})
        ds1 = Dataset({'time': [datetime(1999, 1, 1), datetime(1999, 2, 4)]})
        expected = {(0,): ds1, (1,): ds0}
        actual, concat_dims = _infer_concat_order_from_coords([ds0, ds1])
        assert_combined_tile_ids_equal(expected, actual)
        assert concat_dims == ['time']