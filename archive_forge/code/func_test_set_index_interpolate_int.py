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
@pytest.mark.parametrize('engine', ['pandas', pytest.param('cudf', marks=pytest.mark.gpu)])
def test_set_index_interpolate_int(engine):
    if engine == 'cudf':
        cudf = pytest.importorskip('cudf')
        dask_cudf = pytest.importorskip('dask_cudf')
    L = sorted(list(range(0, 200, 10)) * 2)
    df = pd.DataFrame({'x': 2 * L})
    if engine == 'cudf':
        gdf = cudf.from_pandas(df)
        d = dask_cudf.from_cudf(gdf, npartitions=2)
    else:
        d = dd.from_pandas(df, 2)
    d1 = d.set_index('x', npartitions=10)
    assert all((np.issubdtype(type(x), np.integer) for x in d1.divisions))