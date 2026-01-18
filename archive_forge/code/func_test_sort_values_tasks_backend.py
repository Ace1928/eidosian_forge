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
@pytest.mark.parametrize('backend', ['pandas', pytest.param('cudf', marks=pytest.mark.gpu)])
@pytest.mark.parametrize('by', ['x', 'z', ['x', 'z'], ['z', 'x']])
@pytest.mark.parametrize('ascending', [True, False])
def test_sort_values_tasks_backend(backend, by, ascending):
    if backend == 'cudf':
        pytest.importorskip('dask_cudf')
    pdf = pd.DataFrame({'x': range(10), 'y': [1, 2, 3, 4, 5] * 2, 'z': ['cat', 'dog'] * 5})
    ddf = dd.from_pandas(pdf, npartitions=10)
    if backend == 'cudf':
        ddf = ddf.to_backend(backend)
    expect = pdf.sort_values(by=by, ascending=ascending)
    got = dd.DataFrame.sort_values(ddf, by=by, ascending=ascending, shuffle_method='tasks')
    dd.assert_eq(got, expect, sort_results=False)
    if DASK_EXPR_ENABLED:
        return
    with pytest.warns(FutureWarning, match="'shuffle' keyword is deprecated"):
        got = dd.DataFrame.sort_values(ddf, by=by, ascending=ascending, shuffle='tasks')
    dd.assert_eq(got, expect, sort_results=False)
    with pytest.warns(FutureWarning, match="'shuffle' keyword is deprecated"):
        got = ddf.sort_values(by=by, ascending=ascending, shuffle='tasks')
    dd.assert_eq(got, expect, sort_results=False)