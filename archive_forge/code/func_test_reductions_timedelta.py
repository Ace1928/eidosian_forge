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
@pytest.mark.parametrize('split_every', [False, 2])
def test_reductions_timedelta(split_every):
    ds = pd.Series(pd.to_timedelta([2, 3, 4, np.nan, 5]))
    dds = dd.from_pandas(ds, 2)
    assert_eq(dds.sum(split_every=split_every), ds.sum())
    assert_eq(dds.min(split_every=split_every), ds.min())
    assert_eq(dds.max(split_every=split_every), ds.max())
    assert_eq(dds.count(split_every=split_every), ds.count())