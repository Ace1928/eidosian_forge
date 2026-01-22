from __future__ import annotations
import copy
from datetime import datetime
from typing import Any
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core.indexes import (
from xarray.core.variable import IndexVariable, Variable
from xarray.tests import assert_array_equal, assert_identical, requires_cftime
from xarray.tests.test_coding_times import _all_cftime_date_types
class TestIndexes:

    @pytest.fixture
    def indexes_and_vars(self) -> tuple[list[PandasIndex], dict[Hashable, Variable]]:
        x_idx = PandasIndex(pd.Index([1, 2, 3], name='x'), 'x')
        y_idx = PandasIndex(pd.Index([4, 5, 6], name='y'), 'y')
        z_pd_midx = pd.MultiIndex.from_product([['a', 'b'], [1, 2]], names=['one', 'two'])
        z_midx = PandasMultiIndex(z_pd_midx, 'z')
        indexes = [x_idx, y_idx, z_midx]
        variables = {}
        for idx in indexes:
            variables.update(idx.create_variables())
        return (indexes, variables)

    @pytest.fixture(params=['pd_index', 'xr_index'])
    def unique_indexes(self, request, indexes_and_vars) -> list[PandasIndex] | list[pd.Index]:
        xr_indexes, _ = indexes_and_vars
        if request.param == 'pd_index':
            return [idx.index for idx in xr_indexes]
        else:
            return xr_indexes

    @pytest.fixture
    def indexes(self, unique_indexes, indexes_and_vars) -> Indexes[Index] | Indexes[pd.Index]:
        x_idx, y_idx, z_midx = unique_indexes
        indexes: dict[Any, Index] = {'x': x_idx, 'y': y_idx, 'z': z_midx, 'one': z_midx, 'two': z_midx}
        _, variables = indexes_and_vars
        if isinstance(x_idx, Index):
            index_type = Index
        else:
            index_type = pd.Index
        return Indexes(indexes, variables, index_type=index_type)

    def test_interface(self, unique_indexes, indexes) -> None:
        x_idx = unique_indexes[0]
        assert list(indexes) == ['x', 'y', 'z', 'one', 'two']
        assert len(indexes) == 5
        assert 'x' in indexes
        assert indexes['x'] is x_idx

    def test_variables(self, indexes) -> None:
        assert tuple(indexes.variables) == ('x', 'y', 'z', 'one', 'two')

    def test_dims(self, indexes) -> None:
        assert indexes.dims == {'x': 3, 'y': 3, 'z': 4}

    def test_get_unique(self, unique_indexes, indexes) -> None:
        assert indexes.get_unique() == unique_indexes

    def test_is_multi(self, indexes) -> None:
        assert indexes.is_multi('one') is True
        assert indexes.is_multi('x') is False

    def test_get_all_coords(self, indexes) -> None:
        expected = {'z': indexes.variables['z'], 'one': indexes.variables['one'], 'two': indexes.variables['two']}
        assert indexes.get_all_coords('one') == expected
        with pytest.raises(ValueError, match='errors must be.*'):
            indexes.get_all_coords('x', errors='invalid')
        with pytest.raises(ValueError, match='no index found.*'):
            indexes.get_all_coords('no_coord')
        assert indexes.get_all_coords('no_coord', errors='ignore') == {}

    def test_get_all_dims(self, indexes) -> None:
        expected = {'z': 4}
        assert indexes.get_all_dims('one') == expected

    def test_group_by_index(self, unique_indexes, indexes):
        expected = [(unique_indexes[0], {'x': indexes.variables['x']}), (unique_indexes[1], {'y': indexes.variables['y']}), (unique_indexes[2], {'z': indexes.variables['z'], 'one': indexes.variables['one'], 'two': indexes.variables['two']})]
        assert indexes.group_by_index() == expected

    def test_to_pandas_indexes(self, indexes) -> None:
        pd_indexes = indexes.to_pandas_indexes()
        assert isinstance(pd_indexes, Indexes)
        assert all([isinstance(idx, pd.Index) for idx in pd_indexes.values()])
        assert indexes.variables == pd_indexes.variables

    def test_copy_indexes(self, indexes) -> None:
        copied, index_vars = indexes.copy_indexes()
        assert copied.keys() == indexes.keys()
        for new, original in zip(copied.values(), indexes.values()):
            assert new.equals(original)
        assert copied['z'] is copied['one'] is copied['two']
        assert index_vars.keys() == indexes.variables.keys()
        for new, original in zip(index_vars.values(), indexes.variables.values()):
            assert_identical(new, original)