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
def test_shuffle_aggregate_defaults(shuffle_method):
    pdf = pd.DataFrame({'a': [1, 2, 3, 1, 1, 2, 4, 3, 7] * 100, 'b': [4, 2, 7, 3, 3, 1, 1, 1, 2] * 100, 'c': [0, 1, 2, 3, 4, 5, 6, 7, 8] * 100, 'd': [3, 2, 1, 3, 2, 1, 2, 6, 4] * 100}, columns=['c', 'b', 'a', 'd'])
    ddf = dd.from_pandas(pdf, npartitions=100)
    spec = {'b': 'mean', 'c': ['min', 'max']}
    dsk = ddf.groupby('a').agg(spec, split_out=1).dask
    if not DASK_EXPR_ENABLED:
        assert not any(('shuffle' in l for l in dsk.layers))
    with pytest.raises(ValueError):
        ddf.groupby('a').agg(spec, split_out=1, split_every=1).compute()
    dsk = ddf.groupby('a', sort=False).agg(spec, split_out=2, split_every=1).dask
    if not DASK_EXPR_ENABLED:
        assert any(('shuffle' in l for l in dsk.layers))