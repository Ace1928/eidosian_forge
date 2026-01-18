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
def test_setitem_using_list(self) -> None:
    var1 = Variable(['dim1'], np.random.randn(8))
    var2 = Variable(['dim1'], np.random.randn(8))
    actual = create_test_data()
    expected = actual.copy()
    expected['A'] = var1
    expected['B'] = var2
    actual[['A', 'B']] = [var1, var2]
    assert_identical(actual, expected)
    dv = 2 * expected[['A', 'B']]
    actual[['C', 'D']] = [d.variable for d in dv.data_vars.values()]
    expected[['C', 'D']] = dv
    assert_identical(actual, expected)