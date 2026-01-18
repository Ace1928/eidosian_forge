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
def test_rename_dims(self) -> None:
    original = Dataset({'x': ('x', [0, 1, 2]), 'y': ('x', [10, 11, 12]), 'z': 42})
    expected = Dataset({'x': ('x_new', [0, 1, 2]), 'y': ('x_new', [10, 11, 12]), 'z': 42})
    expected = expected.set_coords('x')
    actual = original.rename_dims({'x': 'x_new'})
    assert_identical(expected, actual, check_default_indexes=False)
    actual_2 = original.rename_dims(x='x_new')
    assert_identical(expected, actual_2, check_default_indexes=False)
    dims_dict_bad = {'x_bad': 'x_new'}
    with pytest.raises(ValueError):
        original.rename_dims(dims_dict_bad)
    with pytest.raises(ValueError):
        original.rename_dims({'x': 'z'})