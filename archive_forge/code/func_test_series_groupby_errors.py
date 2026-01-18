from __future__ import annotations
import contextlib
import operator
import warnings
from datetime import datetime
from functools import partial
import numpy as np
import pandas as pd
import pytest
import dask
import dask.dataframe as dd
from dask.dataframe import _compat
from dask.dataframe._compat import (
from dask.dataframe._pyarrow import to_pyarrow_string
from dask.dataframe.backends import grouper_dispatch
from dask.dataframe.groupby import NUMERIC_ONLY_NOT_IMPLEMENTED
from dask.dataframe.utils import assert_dask_graph, assert_eq, pyarrow_strings_enabled
from dask.utils import M
from dask.utils_test import _check_warning, hlg_layer
def test_series_groupby_errors():
    s = pd.Series([1, 2, 2, 1, 1])
    ss = dd.from_pandas(s, npartitions=2)
    msg = 'No group keys passed!'
    with pytest.raises(ValueError) as err:
        s.groupby([])
    assert msg in str(err.value)
    with pytest.raises(ValueError) as err:
        ss.groupby([])
    assert msg in str(err.value)
    sss = dd.from_pandas(s, npartitions=5)
    with pytest.raises((NotImplementedError, ValueError)):
        ss.groupby(sss)
    with pytest.raises(KeyError):
        s.groupby('x')
    with pytest.raises(KeyError):
        ss.groupby('x')