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
def test_to_stacked_array_to_unstacked_dataset(self) -> None:
    arr = xr.DataArray(np.arange(3), coords=[('x', [0, 1, 2])])
    data = xr.Dataset({'a': arr, 'b': arr})
    stacked = data.to_stacked_array('y', sample_dims=['x'])
    unstacked = stacked.to_unstacked_dataset('y')
    assert_identical(unstacked, data)
    a, b = create_test_stacked_array()
    D = xr.Dataset({'a': a, 'b': b})
    sample_dims = ['x']
    y = D.to_stacked_array('features', sample_dims).transpose('x', 'features')
    x = y.to_unstacked_dataset('features')
    assert_identical(D, x)
    x0 = y[0].to_unstacked_dataset('features')
    d0 = D.isel(x=0)
    assert_identical(d0, x0)