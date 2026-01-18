from __future__ import annotations
import contextlib
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_scalar
import dask.dataframe as dd
from dask.array.numpy_compat import NUMPY_GE_125
from dask.dataframe._compat import (
from dask.dataframe.utils import (
@pytest.mark.skipif(DASK_EXPR_ENABLED, reason='scalar not available like this')
def test_scalar_arithmetics_with_dask_instances():
    s = dd.core.Scalar({('s', 0): 10}, 's', 'i8')
    e = 10
    pds = pd.Series([1, 2, 3, 4, 5, 6, 7])
    dds = dd.from_pandas(pds, 2)
    pdf = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7], 'b': [7, 6, 5, 4, 3, 2, 1]})
    ddf = dd.from_pandas(pdf, 2)
    result = pds + s
    assert isinstance(result, pd.Series)
    assert_eq(result, pds + e)
    result = s + pds
    assert isinstance(result, dd.Series)
    assert_eq(result, pds + e)
    result = dds + s
    assert isinstance(result, dd.Series)
    assert_eq(result, pds + e)
    result = s + dds
    assert isinstance(result, dd.Series)
    assert_eq(result, pds + e)
    result = pdf + s
    assert isinstance(result, pd.DataFrame)
    assert_eq(result, pdf + e)
    result = s + pdf
    assert isinstance(result, dd.DataFrame)
    assert_eq(result, pdf + e)
    result = ddf + s
    assert isinstance(result, dd.DataFrame)
    assert_eq(result, pdf + e)
    result = s + ddf
    assert isinstance(result, dd.DataFrame)
    assert_eq(result, pdf + e)