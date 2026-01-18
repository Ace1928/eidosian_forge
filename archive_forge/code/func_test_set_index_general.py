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
@pytest.mark.parametrize('npartitions', [1, 4, 7, pytest.param(23, marks=pytest.mark.slow)])
def test_set_index_general(npartitions, shuffle_method):
    names = ['alice', 'bob', 'ricky']
    df = pd.DataFrame({'x': np.random.random(100), 'y': np.random.random(100) // 0.2, 'z': np.random.choice(names, 100)}, index=np.random.random(100))
    if PANDAS_GE_140:
        df = df.astype({'x': 'Float64', 'z': 'string'})
    ddf = dd.from_pandas(df, npartitions=npartitions)
    assert_eq(df.set_index('x'), ddf.set_index('x', shuffle_method=shuffle_method))
    assert_eq(df.set_index('y'), ddf.set_index('y', shuffle_method=shuffle_method))
    assert_eq(df.set_index('z'), ddf.set_index('z', shuffle_method=shuffle_method))
    assert_eq(df.set_index(df.x), ddf.set_index(ddf.x, shuffle_method=shuffle_method))
    assert_eq(df.set_index(df.x + df.y), ddf.set_index(ddf.x + ddf.y, shuffle_method=shuffle_method))
    assert_eq(df.set_index(df.x + 1), ddf.set_index(ddf.x + 1, shuffle_method=shuffle_method))
    if DASK_EXPR_ENABLED:
        return
    with pytest.warns(FutureWarning, match="'shuffle' keyword is deprecated"):
        assert_eq(df.set_index('x'), ddf.set_index('x', shuffle=shuffle_method))