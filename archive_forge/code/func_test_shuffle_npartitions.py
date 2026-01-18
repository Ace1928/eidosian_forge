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
def test_shuffle_npartitions(shuffle_method):
    df = pd.DataFrame({'x': np.random.random(100)})
    ddf = dd.from_pandas(df, npartitions=10)
    s = shuffle(ddf, ddf.x, shuffle_method=shuffle_method, npartitions=17, max_branch=4)
    sc = s.compute()
    assert s.npartitions == 17
    assert set(s.dask).issuperset(set(ddf.dask))
    assert len(sc) == len(df)
    assert list(s.columns) == list(df.columns)
    assert set(map(tuple, sc.values.tolist())) == set(map(tuple, df.values.tolist()))