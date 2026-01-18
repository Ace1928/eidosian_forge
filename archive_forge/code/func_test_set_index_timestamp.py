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
def test_set_index_timestamp():
    df = pd.DataFrame({'A': pd.date_range('2000', periods=12, tz='US/Central'), 'B': 1})
    ddf = dd.from_pandas(df, 2)
    divisions = (pd.Timestamp('2000-01-01 00:00:00-0600', tz='US/Central'), pd.Timestamp('2000-01-12 00:00:00-0600', tz='US/Central'))
    df2 = df.set_index('A')
    ddf_new_div = ddf.set_index('A', divisions=divisions)
    for ts1, ts2 in zip(divisions, ddf_new_div.divisions):
        assert ts1.timetuple() == ts2.timetuple()
        assert ts1.tz == ts2.tz
    assert_eq(df2, ddf_new_div, check_freq=False)
    assert_eq(df2, ddf.set_index('A'), check_freq=False)