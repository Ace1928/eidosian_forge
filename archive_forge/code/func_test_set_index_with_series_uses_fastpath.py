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
def test_set_index_with_series_uses_fastpath():
    dates = pd.date_range(start='2022-02-22', freq='16h', periods=12) - pd.Timedelta(seconds=30)
    one_day = pd.Timedelta(days=1)
    df = pd.DataFrame({'x': [1, 2, 3, 4] * 3, 'y': [10, 20, 30] * 4, 'name': ['Alice', 'Bob'] * 6, 'd1': dates + one_day, 'd2': dates + one_day * 5}, index=dates)
    ddf = dd.from_pandas(df, npartitions=3)
    res = ddf.set_index(ddf.d2 + one_day)
    expected = df.set_index(df.d2 + one_day)
    assert_eq(res, expected)