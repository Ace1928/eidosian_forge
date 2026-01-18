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
def test_series_groupby():
    s = pd.Series([1, 2, 2, 1, 1])
    pd_group = s.groupby(s)
    ss = dd.from_pandas(s, npartitions=2)
    dask_group = ss.groupby(ss)
    pd_group2 = s.groupby(s + 1)
    dask_group2 = ss.groupby(ss + 1)
    for dg, pdg in [(dask_group, pd_group), (pd_group2, dask_group2)]:
        assert_eq(dg.count(), pdg.count())
        assert_eq(dg.sum(), pdg.sum())
        assert_eq(dg.min(), pdg.min())
        assert_eq(dg.max(), pdg.max())
        assert_eq(dg.size(), pdg.size())
        assert_eq(dg.first(), pdg.first())
        assert_eq(dg.last(), pdg.last())
        assert_eq(dg.prod(), pdg.prod())