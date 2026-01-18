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
def test_set_index_names(shuffle_method):
    if shuffle_method == 'disk':
        pytest.xfail('dsk names in disk shuffle are not deterministic')
    df = pd.DataFrame({'x': np.random.random(100), 'y': np.random.random(100) // 0.2}, index=np.random.random(100))
    ddf = dd.from_pandas(df, npartitions=4)
    assert set(ddf.set_index('x', shuffle_method=shuffle_method).dask) == set(ddf.set_index('x', shuffle_method=shuffle_method).dask)
    assert set(ddf.set_index('x', shuffle_method=shuffle_method).dask) != set(ddf.set_index('y', shuffle_method=shuffle_method).dask)
    assert set(ddf.set_index('x', max_branch=4, shuffle_method=shuffle_method).dask) != set(ddf.set_index('x', max_branch=3, shuffle_method=shuffle_method).dask)
    assert set(ddf.set_index('x', drop=True, shuffle_method=shuffle_method).dask) != set(ddf.set_index('x', drop=False, shuffle_method=shuffle_method).dask)