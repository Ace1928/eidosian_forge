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
def test_stack_non_dim_coords(self) -> None:
    ds = Dataset(data_vars={'b': (('x', 'y'), [[0, 1], [2, 3]])}, coords={'x': ('x', [0, 1]), 'y': ['a', 'b']}).rename_vars(x='xx')
    exp_index = pd.MultiIndex.from_product([[0, 1], ['a', 'b']], names=['xx', 'y'])
    exp_coords = Coordinates.from_pandas_multiindex(exp_index, 'z')
    expected = Dataset(data_vars={'b': ('z', [0, 1, 2, 3])}, coords=exp_coords)
    actual = ds.stack(z=['x', 'y'])
    assert_identical(expected, actual)
    assert list(actual.xindexes) == ['z', 'xx', 'y']