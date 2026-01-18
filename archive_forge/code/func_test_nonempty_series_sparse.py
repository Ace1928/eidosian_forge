from __future__ import annotations
import re
import warnings
from collections.abc import Iterable
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
import dask
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_200, PANDAS_GE_300, tm
from dask.dataframe.core import apply_and_enforce
from dask.dataframe.utils import (
from dask.local import get_sync
def test_nonempty_series_sparse():
    ser = pd.Series(pd.array([0, 1], dtype='Sparse'))
    with warnings.catch_warnings(record=True) as record:
        meta_nonempty(ser)
    assert not record