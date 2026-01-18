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
@pytest.mark.parametrize('func', [lambda x: x.clip(0, 1), lambda x: np.float64(1.0) * x, np.abs, abs])
def test_propagate_attrs(self, func) -> None:
    da = DataArray(range(5), name='a', attrs={'attr': 'da'})
    ds = Dataset({'a': da}, attrs={'attr': 'ds'})
    assert func(ds).attrs == ds.attrs
    with set_options(keep_attrs=False):
        assert func(ds).attrs != ds.attrs
        assert func(ds).a.attrs != ds.a.attrs
    with set_options(keep_attrs=False):
        assert func(ds).attrs != ds.attrs
        assert func(ds).a.attrs != ds.a.attrs
    with set_options(keep_attrs=True):
        assert func(ds).attrs == ds.attrs
        assert func(ds).a.attrs == ds.a.attrs