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
@pytest.mark.parametrize('converter', [int, float, str, lambda x: pd.to_datetime(x, unit='ns')])
def test_set_index_on_empty(converter):
    test_vals = [1, 2, 3, 4]
    df = pd.DataFrame([{'x': converter(x), 'y': x} for x in test_vals])
    ddf = dd.from_pandas(df, npartitions=4)
    assert ddf.npartitions > 1
    actual = ddf[ddf.y > df.y.max()].set_index('x')
    expected = df[df.y > df.y.max()].set_index('x')
    assert assert_eq(actual, expected, check_freq=False)
    assert actual.npartitions == 1
    assert all((pd.isnull(d) for d in actual.divisions))
    if not DASK_EXPR_ENABLED:
        actual = ddf[ddf.y > df.y.max()].set_index('x', sorted=True)
        assert assert_eq(actual, expected, check_freq=False)
        assert actual.npartitions == 1
        assert all((pd.isnull(d) for d in actual.divisions))