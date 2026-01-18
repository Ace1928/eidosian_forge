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
def test_reduce_non_numeric(self) -> None:
    data1 = create_test_data(seed=44)
    data2 = create_test_data(seed=44)
    add_vars = {'var4': ['dim1', 'dim2'], 'var5': ['dim1']}
    for v, dims in sorted(add_vars.items()):
        size = tuple((data1.sizes[d] for d in dims))
        data = np.random.randint(0, 100, size=size).astype(np.str_)
        data1[v] = (dims, data, {'foo': 'variable'})
    assert 'var4' not in data1.mean() and 'var5' not in data1.mean()
    assert_equal(data1.mean(), data2.mean())
    assert_equal(data1.mean(dim='dim1'), data2.mean(dim='dim1'))
    assert 'var4' not in data1.mean(dim='dim2') and 'var5' in data1.mean(dim='dim2')