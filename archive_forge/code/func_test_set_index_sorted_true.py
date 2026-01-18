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
def test_set_index_sorted_true():
    df = pd.DataFrame({'x': [1, 2, 3, 4], 'y': [10, 20, 20, 40], 'z': [4, 3, 2, 1]})
    a = dd.from_pandas(df, 2, sort=False).clear_divisions()
    assert not a.known_divisions
    b = a.set_index('x', sorted=True)
    assert b.known_divisions
    assert set(a.dask).issubset(set(b.dask))
    for drop in [True, False]:
        assert_eq(a.set_index('x', drop=drop), df.set_index('x', drop=drop))
        assert_eq(a.set_index(a.x, sorted=True, drop=drop), df.set_index(df.x, drop=drop))
        assert_eq(a.set_index(a.x + 1, sorted=True, drop=drop), df.set_index(df.x + 1, drop=drop))
    if not DASK_EXPR_ENABLED:
        with pytest.raises(ValueError):
            a.set_index(a.z, sorted=True)