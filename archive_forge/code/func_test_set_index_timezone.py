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
@pytest.mark.skipif(DASK_EXPR_ENABLED, reason="we don't do division inference for 1 partition frames")
def test_set_index_timezone():
    s_naive = pd.Series(pd.date_range('20130101', periods=3))
    s_aware = pd.Series(pd.date_range('20130101', periods=3, tz='US/Eastern'))
    df = pd.DataFrame({'tz': s_aware, 'notz': s_naive})
    d = dd.from_pandas(df, npartitions=1)
    d1 = d.set_index('notz', npartitions=1)
    s1 = pd.DatetimeIndex(s_naive.values, dtype=s_naive.dtype)
    assert d1.divisions[0] == s_naive[0] == s1[0]
    assert d1.divisions[-1] == s_naive[2] == s1[2]
    d2 = d.set_index('tz', npartitions=1)
    s2 = pd.DatetimeIndex(s_aware, dtype=s_aware.dtype)
    assert d2.divisions[0] == s2[0]
    assert d2.divisions[-1] == s2[2]
    assert d2.divisions[0].tz == s2[0].tz
    assert d2.divisions[0].tz is not None
    s2badtype = pd.DatetimeIndex(s_aware.values, dtype=s_naive.dtype)
    assert not d2.divisions[0] == s2badtype[0]