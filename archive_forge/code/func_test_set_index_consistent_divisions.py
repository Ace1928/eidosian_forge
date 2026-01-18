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
@pytest.mark.slow
def test_set_index_consistent_divisions():
    df = pd.DataFrame({'x': np.random.random(100), 'y': np.random.random(100) // 0.2}, index=np.random.random(100))
    ddf = dd.from_pandas(df, npartitions=4)
    ddf = ddf.clear_divisions()
    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(8, ctx) as pool:
        func = partial(_set_index, df=ddf, idx='x')
        divisions_set = set(pool.map(func, range(100)))
    assert len(divisions_set) == 1