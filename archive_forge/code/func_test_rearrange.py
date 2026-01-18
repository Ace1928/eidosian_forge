from __future__ import annotations
import contextlib
import itertools
import multiprocessing as mp
import os
import pickle
import random
import string
import tempfile
from concurrent.futures import ProcessPoolExecutor
from copy import copy
from datetime import date, time
from decimal import Decimal
from functools import partial
from unittest import mock
import numpy as np
import pandas as pd
import pytest
import dask
import dask.dataframe as dd
from dask.base import compute_as_if_collection
from dask.dataframe._compat import (
from dask.dataframe.shuffle import (
from dask.dataframe.utils import assert_eq, make_meta
from dask.optimization import cull
@pytest.mark.skipif(DASK_EXPR_ENABLED, reason='not available')
@pytest.mark.parametrize('scheduler', ['threads', 'processes'])
def test_rearrange(shuffle_method, scheduler):
    df = pd.DataFrame({'x': np.random.random(10)})
    ddf = dd.from_pandas(df, npartitions=4)
    ddf2 = ddf.assign(_partitions=ddf.x % 4)
    result = rearrange_by_column(ddf2, '_partitions', max_branch=32, shuffle_method=shuffle_method)
    assert result.npartitions == ddf.npartitions
    assert set(ddf.dask).issubset(result.dask)
    a = result.compute(scheduler=scheduler)
    get = dask.base.get_scheduler(scheduler=scheduler)
    parts = get(result.dask, result.__dask_keys__())
    for i in a._partitions.drop_duplicates():
        assert sum((i in set(part._partitions) for part in parts)) == 1