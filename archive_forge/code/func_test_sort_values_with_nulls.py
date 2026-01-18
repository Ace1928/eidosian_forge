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
@pytest.mark.parametrize('na_position', ['first', 'last'])
@pytest.mark.parametrize('ascending', [True, False])
@pytest.mark.parametrize('by', ['a', 'b', ['a', 'b']])
@pytest.mark.parametrize('nparts', [1, 5])
@pytest.mark.parametrize('data', [{'a': list(range(50)) + [None] * 50 + list(range(50, 100)), 'b': [None] * 100 + list(range(100, 150))}, {'a': list(range(15)) + [None] * 5, 'b': list(reversed(range(20)))}])
def test_sort_values_with_nulls(data, nparts, by, ascending, na_position):
    df = pd.DataFrame(data)
    ddf = dd.from_pandas(df, npartitions=nparts)
    with dask.config.set(scheduler='single-threaded'):
        got = ddf.sort_values(by=by, ascending=ascending, na_position=na_position)
    expect = df.sort_values(by=by, ascending=ascending, na_position=na_position)
    dd.assert_eq(got, expect, check_index=False)