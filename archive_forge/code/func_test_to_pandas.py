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
def test_to_pandas(self) -> None:
    actual = Dataset({'a': 1, 'b': 2}).to_pandas()
    expected = pd.Series([1, 2], ['a', 'b'])
    assert_array_equal(actual, expected)
    x = np.random.randn(10)
    y = np.random.randn(10)
    t = list('abcdefghij')
    ds = Dataset({'a': ('t', x), 'b': ('t', y), 't': ('t', t)})
    actual = ds.to_pandas()
    expected = ds.to_dataframe()
    assert expected.equals(actual), (expected, actual)
    x2d = np.random.randn(10, 10)
    y2d = np.random.randn(10, 10)
    with pytest.raises(ValueError, match='cannot convert Datasets'):
        Dataset({'a': (['t', 'r'], x2d), 'b': (['t', 'r'], y2d)}).to_pandas()