from __future__ import annotations
import warnings
from abc import ABC
from copy import copy, deepcopy
from datetime import datetime, timedelta
from textwrap import dedent
from typing import Generic
import numpy as np
import pandas as pd
import pytest
import pytz
from xarray import DataArray, Dataset, IndexVariable, Variable, set_options
from xarray.core import dtypes, duck_array_ops, indexing
from xarray.core.common import full_like, ones_like, zeros_like
from xarray.core.indexing import (
from xarray.core.types import T_DuckArray
from xarray.core.utils import NDArrayMixin
from xarray.core.variable import as_compatible_data, as_variable
from xarray.namedarray.pycompat import array_type
from xarray.tests import (
from xarray.tests.test_namedarray import NamedArraySubclassobjects
@requires_pandas_version_two
def test_pandas_two_only_datetime_conversion_warnings() -> None:
    cases = [(pd.date_range('2000', periods=1), 'datetime64[s]'), (pd.Series(pd.date_range('2000', periods=1)), 'datetime64[s]'), (pd.date_range('2000', periods=1, tz=pytz.timezone('America/New_York')), pd.DatetimeTZDtype('s', pytz.timezone('America/New_York'))), (pd.Series(pd.date_range('2000', periods=1, tz=pytz.timezone('America/New_York'))), pd.DatetimeTZDtype('s', pytz.timezone('America/New_York')))]
    for data, dtype in cases:
        with pytest.warns(UserWarning, match='non-nanosecond precision datetime'):
            var = Variable(['time'], data.astype(dtype))
    if var.dtype.kind == 'M':
        assert var.dtype == np.dtype('datetime64[ns]')
    else:
        assert isinstance(var._data, PandasIndexingAdapter)
        assert var._data.array.dtype == pd.DatetimeTZDtype('ns', pytz.timezone('America/New_York'))