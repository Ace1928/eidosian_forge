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
def test_series_groupby_multi_character_column_name():
    df = pd.DataFrame({'aa': [1, 2, 1, 3, 4, 1, 2]})
    ddf = dd.from_pandas(df, npartitions=3)
    assert_eq(df.groupby('aa').aa.cumsum(), ddf.groupby('aa').aa.cumsum())