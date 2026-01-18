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
def test_set_xindex(self) -> None:
    ds = Dataset(coords={'foo': ('x', ['a', 'a', 'b', 'b']), 'bar': ('x', [0, 1, 2, 3])})
    actual = ds.set_xindex('foo')
    expected = ds.set_index(x='foo').rename_vars(x='foo')
    assert_identical(actual, expected, check_default_indexes=False)
    actual_mindex = ds.set_xindex(['foo', 'bar'])
    expected_mindex = ds.set_index(x=['foo', 'bar'])
    assert_identical(actual_mindex, expected_mindex)

    class NotAnIndex:
        ...
    with pytest.raises(TypeError, match='.*not a subclass of xarray.Index'):
        ds.set_xindex('foo', NotAnIndex)
    with pytest.raises(ValueError, match="those variables don't exist"):
        ds.set_xindex('not_a_coordinate', PandasIndex)
    ds['data_var'] = ('x', [1, 2, 3, 4])
    with pytest.raises(ValueError, match='those variables are data variables'):
        ds.set_xindex('data_var', PandasIndex)
    ds2 = Dataset(coords={'x': ('x', [0, 1, 2, 3])})
    with pytest.raises(ValueError, match='those coordinates already have an index'):
        ds2.set_xindex('x', PandasIndex)