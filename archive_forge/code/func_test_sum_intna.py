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
def test_sum_intna():
    a = pd.Series([1, None, 2], dtype=pd.Int32Dtype())
    b = dd.from_pandas(a, 2)
    assert_eq(a.sum(), b.sum())