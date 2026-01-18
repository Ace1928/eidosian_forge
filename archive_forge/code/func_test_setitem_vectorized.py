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
def test_setitem_vectorized(self) -> None:
    da = xr.DataArray(np.r_[:120].reshape(2, 3, 4, 5), dims=['a', 'b', 'c', 'd'])
    ds = xr.Dataset({'da': da})
    b = xr.DataArray([[0, 0], [1, 0]], dims=['u', 'v'])
    c = xr.DataArray([[0, 1], [2, 3]], dims=['u', 'v'])
    w = xr.DataArray([-1, -2], dims=['u'])
    index = dict(b=b, c=c)
    ds[index] = xr.Dataset({'da': w})
    assert (ds[index]['da'] == w).all()
    da = xr.DataArray(np.r_[:120].reshape(2, 3, 4, 5), dims=['a', 'b', 'c', 'd'])
    ds = xr.Dataset({'da': da})
    ds.coords['b'] = [2, 4, 6]
    b = xr.DataArray([[2, 2], [4, 2]], dims=['u', 'v'])
    c = xr.DataArray([[0, 1], [2, 3]], dims=['u', 'v'])
    w = xr.DataArray([-1, -2], dims=['u'])
    index = dict(b=b, c=c)
    ds.loc[index] = xr.Dataset({'da': w}, coords={'b': ds.coords['b']})
    assert (ds.loc[index]['da'] == w).all()