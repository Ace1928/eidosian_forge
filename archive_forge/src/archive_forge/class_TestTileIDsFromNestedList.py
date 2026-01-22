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
class TestTileIDsFromNestedList:

    def test_1d(self):
        ds = create_test_data
        input = [ds(0), ds(1)]
        expected = {(0,): ds(0), (1,): ds(1)}
        actual = _infer_concat_order_from_positions(input)
        assert_combined_tile_ids_equal(expected, actual)

    def test_2d(self):
        ds = create_test_data
        input = [[ds(0), ds(1)], [ds(2), ds(3)], [ds(4), ds(5)]]
        expected = {(0, 0): ds(0), (0, 1): ds(1), (1, 0): ds(2), (1, 1): ds(3), (2, 0): ds(4), (2, 1): ds(5)}
        actual = _infer_concat_order_from_positions(input)
        assert_combined_tile_ids_equal(expected, actual)

    def test_3d(self):
        ds = create_test_data
        input = [[[ds(0), ds(1)], [ds(2), ds(3)], [ds(4), ds(5)]], [[ds(6), ds(7)], [ds(8), ds(9)], [ds(10), ds(11)]]]
        expected = {(0, 0, 0): ds(0), (0, 0, 1): ds(1), (0, 1, 0): ds(2), (0, 1, 1): ds(3), (0, 2, 0): ds(4), (0, 2, 1): ds(5), (1, 0, 0): ds(6), (1, 0, 1): ds(7), (1, 1, 0): ds(8), (1, 1, 1): ds(9), (1, 2, 0): ds(10), (1, 2, 1): ds(11)}
        actual = _infer_concat_order_from_positions(input)
        assert_combined_tile_ids_equal(expected, actual)

    def test_single_dataset(self):
        ds = create_test_data(0)
        input = [ds]
        expected = {(0,): ds}
        actual = _infer_concat_order_from_positions(input)
        assert_combined_tile_ids_equal(expected, actual)

    def test_redundant_nesting(self):
        ds = create_test_data
        input = [[ds(0)], [ds(1)]]
        expected = {(0, 0): ds(0), (1, 0): ds(1)}
        actual = _infer_concat_order_from_positions(input)
        assert_combined_tile_ids_equal(expected, actual)

    def test_ignore_empty_list(self):
        ds = create_test_data(0)
        input = [ds, []]
        expected = {(0,): ds}
        actual = _infer_concat_order_from_positions(input)
        assert_combined_tile_ids_equal(expected, actual)

    def test_uneven_depth_input(self):
        ds = create_test_data
        input = [ds(0), [ds(1), ds(2)]]
        expected = {(0,): ds(0), (1, 0): ds(1), (1, 1): ds(2)}
        actual = _infer_concat_order_from_positions(input)
        assert_combined_tile_ids_equal(expected, actual)

    def test_uneven_length_input(self):
        ds = create_test_data
        input = [[ds(0)], [ds(1), ds(2)]]
        expected = {(0, 0): ds(0), (1, 0): ds(1), (1, 1): ds(2)}
        actual = _infer_concat_order_from_positions(input)
        assert_combined_tile_ids_equal(expected, actual)

    def test_infer_from_datasets(self):
        ds = create_test_data
        input = [ds(0), ds(1)]
        expected = {(0,): ds(0), (1,): ds(1)}
        actual = _infer_concat_order_from_positions(input)
        assert_combined_tile_ids_equal(expected, actual)