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
@pytest.mark.parametrize('npartitions', [1, 3])
def test_sort_values_timestamp(npartitions):
    df = pd.DataFrame.from_records([[pd.Timestamp('2002-01-11 21:00:01+0000', tz='UTC'), 4223, 54719.0], [pd.Timestamp('2002-01-14 21:00:01+0000', tz='UTC'), 6942, 19223.0], [pd.Timestamp('2002-01-15 21:00:01+0000', tz='UTC'), 12551, 72865.0], [pd.Timestamp('2002-01-23 21:00:01+0000', tz='UTC'), 6005, 57670.0], [pd.Timestamp('2002-01-29 21:00:01+0000', tz='UTC'), 2043, 58600.0], [pd.Timestamp('2002-02-01 21:00:01+0000', tz='UTC'), 6909, 8459.0], [pd.Timestamp('2002-01-14 21:00:01+0000', tz='UTC'), 5326, 77339.0], [pd.Timestamp('2002-01-14 21:00:01+0000', tz='UTC'), 4711, 54135.0], [pd.Timestamp('2002-01-22 21:00:01+0000', tz='UTC'), 103, 57627.0], [pd.Timestamp('2002-01-30 21:00:01+0000', tz='UTC'), 16862, 54458.0], [pd.Timestamp('2002-01-31 21:00:01+0000', tz='UTC'), 4143, 56280.0]], columns=['time', 'id1', 'id2'])
    ddf = dd.from_pandas(df, npartitions=npartitions)
    result = ddf.sort_values('time')
    expected = df.sort_values('time')
    assert_eq(result, expected)