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
@pytest.mark.parametrize('compute_backend', ['numbagg', None], indirect=True)
@pytest.mark.parametrize('skipna', [True, False])
def test_quantile_skipna(self, skipna, compute_backend) -> None:
    q = 0.1
    dim = 'time'
    ds = Dataset({'a': ([dim], np.arange(0, 11))})
    ds = ds.where(ds >= 1)
    result = ds.quantile(q=q, dim=dim, skipna=skipna)
    value = 1.9 if skipna else np.nan
    expected = Dataset({'a': value}, coords={'quantile': q})
    assert_identical(result, expected)