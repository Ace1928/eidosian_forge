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
def test_sel_method(self) -> None:
    data = create_test_data()
    expected = data.sel(dim2=1)
    actual = data.sel(dim2=0.95, method='nearest')
    assert_identical(expected, actual)
    actual = data.sel(dim2=0.95, method='nearest', tolerance=1)
    assert_identical(expected, actual)
    with pytest.raises(KeyError):
        actual = data.sel(dim2=np.pi, method='nearest', tolerance=0)
    expected = data.sel(dim2=[1.5])
    actual = data.sel(dim2=[1.45], method='backfill')
    assert_identical(expected, actual)
    with pytest.raises(NotImplementedError, match='slice objects'):
        data.sel(dim2=slice(1, 3), method='ffill')
    with pytest.raises(TypeError, match='``method``'):
        data.sel(dim2=1, method=data)
    with pytest.raises(ValueError, match='cannot supply'):
        data.sel(dim1=0, method='nearest')