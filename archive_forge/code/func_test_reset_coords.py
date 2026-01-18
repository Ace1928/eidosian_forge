from __future__ import annotations
import pickle
import re
import sys
import warnings
from collections.abc import Hashable
from copy import deepcopy
from textwrap import dedent
from typing import Any, Final, Literal, cast
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import (
from xarray.coding.times import CFDatetimeCoder
from xarray.core import dtypes
from xarray.core.common import full_like
from xarray.core.coordinates import Coordinates
from xarray.core.indexes import Index, PandasIndex, filter_indexes_from_coords
from xarray.core.types import QueryEngineOptions, QueryParserOptions
from xarray.core.utils import is_scalar
from xarray.testing import _assert_internal_invariants
from xarray.tests import (
def test_reset_coords(self) -> None:
    data = DataArray(np.zeros((3, 4)), {'bar': ('x', ['a', 'b', 'c']), 'baz': ('y', range(4)), 'y': range(4)}, dims=['x', 'y'], name='foo')
    actual1 = data.reset_coords()
    expected1 = Dataset({'foo': (['x', 'y'], np.zeros((3, 4))), 'bar': ('x', ['a', 'b', 'c']), 'baz': ('y', range(4)), 'y': range(4)})
    assert_identical(actual1, expected1)
    actual2 = data.reset_coords(['bar', 'baz'])
    assert_identical(actual2, expected1)
    actual3 = data.reset_coords('bar')
    expected3 = Dataset({'foo': (['x', 'y'], np.zeros((3, 4))), 'bar': ('x', ['a', 'b', 'c'])}, {'baz': ('y', range(4)), 'y': range(4)})
    assert_identical(actual3, expected3)
    actual4 = data.reset_coords(['bar'])
    assert_identical(actual4, expected3)
    actual5 = data.reset_coords(drop=True)
    expected5 = DataArray(np.zeros((3, 4)), coords={'y': range(4)}, dims=['x', 'y'], name='foo')
    assert_identical(actual5, expected5)
    actual6 = data.copy().reset_coords(drop=True)
    assert_identical(actual6, expected5)
    actual7 = data.reset_coords('bar', drop=True)
    expected7 = DataArray(np.zeros((3, 4)), {'baz': ('y', range(4)), 'y': range(4)}, dims=['x', 'y'], name='foo')
    assert_identical(actual7, expected7)
    with pytest.raises(ValueError, match='cannot be found'):
        data.reset_coords('foo', drop=True)
    with pytest.raises(ValueError, match='cannot be found'):
        data.reset_coords('not_found')
    with pytest.raises(ValueError, match='cannot remove index'):
        data.reset_coords('y')
    midx = pd.MultiIndex.from_product([['a', 'b'], [0, 1]], names=('lvl1', 'lvl2'))
    data = DataArray([1, 2, 3, 4], coords={'x': midx}, dims='x', name='foo')
    with pytest.raises(ValueError, match='cannot remove index'):
        data.reset_coords('lvl1')