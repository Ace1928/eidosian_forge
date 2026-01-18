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
def test_shuffle_nulls_introduced():
    df1 = pd.DataFrame([[True], [False]] * 50, columns=['A'])
    df1['B'] = list(range(100))
    df2 = pd.DataFrame([[2, 3], [109, 2], [345, 3], [50, 7], [95, 1]], columns=['B', 'C'])
    ddf1 = dd.from_pandas(df1, npartitions=10)
    ddf2 = dd.from_pandas(df2, npartitions=1)
    meta = pd.Series(dtype=int, index=pd.Index([], dtype=bool, name='A'), name='A')
    include_groups = {'include_groups': False} if PANDAS_GE_220 else {}
    result = dd.merge(ddf1, ddf2, how='outer', on='B').groupby('A').apply(lambda df: len(df), meta=meta, **include_groups)
    expected = pd.merge(df1, df2, how='outer', on='B').groupby('A').apply(lambda df: len(df), **include_groups)
    assert_eq(result, expected, check_names=False)