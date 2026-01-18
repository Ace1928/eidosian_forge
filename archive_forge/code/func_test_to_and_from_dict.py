from __future__ import annotations
import pickle
import re
import sys
import warnings
from collections.abc import Hashable
from copy import copy, deepcopy
from io import StringIO
from textwrap import dedent
from typing import Any, Literal
import numpy as np
import pandas as pd
import pytest
from pandas.core.indexes.datetimes import DatetimeIndex
import xarray as xr
from xarray import (
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core import dtypes, indexing, utils
from xarray.core.common import duck_array_ops, full_like
from xarray.core.coordinates import Coordinates, DatasetCoordinates
from xarray.core.indexes import Index, PandasIndex
from xarray.core.utils import is_scalar
from xarray.namedarray.pycompat import array_type, integer_types
from xarray.testing import _assert_internal_invariants
from xarray.tests import (
@pytest.mark.parametrize('encoding', [True, False])
@pytest.mark.parametrize('data', [True, 'list', 'array'])
def test_to_and_from_dict(self, encoding: bool, data: bool | Literal['list', 'array']) -> None:
    x = np.random.randn(10)
    y = np.random.randn(10)
    t = list('abcdefghij')
    ds = Dataset({'a': ('t', x), 'b': ('t', y), 't': ('t', t)})
    expected: dict[str, dict[str, Any]] = {'coords': {'t': {'dims': ('t',), 'data': t, 'attrs': {}}}, 'attrs': {}, 'dims': {'t': 10}, 'data_vars': {'a': {'dims': ('t',), 'data': x.tolist(), 'attrs': {}}, 'b': {'dims': ('t',), 'data': y.tolist(), 'attrs': {}}}}
    if encoding:
        ds.t.encoding.update({'foo': 'bar'})
        expected['encoding'] = {}
        expected['coords']['t']['encoding'] = ds.t.encoding
        for vvs in ['a', 'b']:
            expected['data_vars'][vvs]['encoding'] = {}
    actual = ds.to_dict(data=data, encoding=encoding)
    np.testing.assert_equal(expected, actual)
    ds_rt = Dataset.from_dict(actual)
    assert_identical(ds, ds_rt)
    if encoding:
        assert set(ds_rt.variables) == set(ds.variables)
        for vv in ds.variables:
            np.testing.assert_equal(ds_rt[vv].encoding, ds[vv].encoding)
    expected_no_data = expected.copy()
    del expected_no_data['coords']['t']['data']
    del expected_no_data['data_vars']['a']['data']
    del expected_no_data['data_vars']['b']['data']
    endiantype = '<U1' if sys.byteorder == 'little' else '>U1'
    expected_no_data['coords']['t'].update({'dtype': endiantype, 'shape': (10,)})
    expected_no_data['data_vars']['a'].update({'dtype': 'float64', 'shape': (10,)})
    expected_no_data['data_vars']['b'].update({'dtype': 'float64', 'shape': (10,)})
    actual_no_data = ds.to_dict(data=False, encoding=encoding)
    assert expected_no_data == actual_no_data
    expected_ds = ds.set_coords('b')
    actual2 = Dataset.from_dict(expected_ds.to_dict(data=data, encoding=encoding))
    assert_identical(expected_ds, actual2)
    if encoding:
        assert set(expected_ds.variables) == set(actual2.variables)
        for vv in ds.variables:
            np.testing.assert_equal(expected_ds[vv].encoding, actual2[vv].encoding)
    d = {'coords': {'t': {'dims': 't', 'data': t}}, 'dims': 't', 'data_vars': {'a': {'dims': 't', 'data': x}, 'b': {'dims': 't', 'data': y}}}
    assert_identical(ds, Dataset.from_dict(d))
    d = {'a': {'dims': 't', 'data': x}, 't': {'data': t, 'dims': 't'}, 'b': {'dims': 't', 'data': y}}
    assert_identical(ds, Dataset.from_dict(d))
    d = {'a': {'data': x}, 't': {'data': t, 'dims': 't'}, 'b': {'dims': 't', 'data': y}}
    with pytest.raises(ValueError, match="cannot convert dict without the key 'dims'"):
        Dataset.from_dict(d)