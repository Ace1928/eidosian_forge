from __future__ import annotations
import contextlib
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_scalar
import dask.dataframe as dd
from dask.array.numpy_compat import NUMPY_GE_125
from dask.dataframe._compat import (
from dask.dataframe.utils import (
@pytest.mark.parametrize('split_every', [False, 2])
def test_reductions_frame(split_every):
    dsk = {('x', 0): pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}, index=[0, 1, 3]), ('x', 1): pd.DataFrame({'a': [4, 5, 6], 'b': [3, 2, 1]}, index=[5, 6, 8]), ('x', 2): pd.DataFrame({'a': [7, 8, 9], 'b': [0, 0, 0]}, index=[9, 9, 9])}
    meta = make_meta({'a': 'i8', 'b': 'i8'}, index=pd.Index([], 'i8'), parent_meta=pd.DataFrame())
    if DASK_EXPR_ENABLED:
        ddf1 = dd.repartition(pd.concat(dsk.values()), [0, 4, 9, 9])
    else:
        ddf1 = dd.DataFrame(dsk, 'x', meta, [0, 4, 9, 9])
    pdf1 = ddf1.compute()
    assert_eq(ddf1.sum(split_every=split_every), pdf1.sum())
    assert_eq(ddf1.prod(split_every=split_every), pdf1.prod())
    assert_eq(ddf1.product(split_every=split_every), pdf1.product())
    assert_eq(ddf1.min(split_every=split_every), pdf1.min())
    assert_eq(ddf1.max(split_every=split_every), pdf1.max())
    assert_eq(ddf1.count(split_every=split_every), pdf1.count())
    assert_eq(ddf1.std(split_every=split_every), pdf1.std())
    assert_eq(ddf1.var(split_every=split_every), pdf1.var())
    assert_eq(ddf1.sem(split_every=split_every), pdf1.sem())
    assert_eq(ddf1.std(ddof=0, split_every=split_every), pdf1.std(ddof=0))
    assert_eq(ddf1.var(ddof=0, split_every=split_every), pdf1.var(ddof=0))
    assert_eq(ddf1.sem(ddof=0, split_every=split_every), pdf1.sem(ddof=0))
    assert_eq(ddf1.mean(split_every=split_every), pdf1.mean())
    for axis in [0, 1, 'index', 'columns']:
        assert_eq(ddf1.sum(axis=axis, split_every=split_every), pdf1.sum(axis=axis))
        assert_eq(ddf1.prod(axis=axis, split_every=split_every), pdf1.prod(axis=axis))
        assert_eq(ddf1.product(axis=axis, split_every=split_every), pdf1.product(axis=axis))
        assert_eq(ddf1.min(axis=axis, split_every=split_every), pdf1.min(axis=axis))
        assert_eq(ddf1.max(axis=axis, split_every=split_every), pdf1.max(axis=axis))
        assert_eq(ddf1.count(axis=axis, split_every=split_every), pdf1.count(axis=axis))
        assert_eq(ddf1.std(axis=axis, split_every=split_every), pdf1.std(axis=axis))
        assert_eq(ddf1.var(axis=axis, split_every=split_every), pdf1.var(axis=axis))
        assert_eq(ddf1.sem(axis=axis, split_every=split_every), pdf1.sem(axis=axis))
        assert_eq(ddf1.std(axis=axis, ddof=0, split_every=split_every), pdf1.std(axis=axis, ddof=0))
        assert_eq(ddf1.var(axis=axis, ddof=0, split_every=split_every), pdf1.var(axis=axis, ddof=0))
        assert_eq(ddf1.sem(axis=axis, ddof=0, split_every=split_every), pdf1.sem(axis=axis, ddof=0))
        assert_eq(ddf1.mean(axis=axis, split_every=split_every), pdf1.mean(axis=axis))
    pytest.raises(ValueError, lambda: ddf1.sum(axis='incorrect').compute())
    if PANDAS_GE_140 and (not PANDAS_GE_200):
        ctx = pytest.warns(FutureWarning, match='axis=None')
    else:
        ctx = contextlib.nullcontext()
    with ctx:
        result = ddf1.min(axis=None, split_every=split_every)
    with ctx:
        expected = pdf1.min(axis=None)
    assert_eq(result, expected)
    with ctx:
        result = ddf1.max(axis=None, split_every=split_every)
    with ctx:
        expected = pdf1.max(axis=None)
    assert_eq(result, expected)
    with ctx:
        result = ddf1.mean(axis=None, split_every=split_every)
    with ctx:
        expected = pdf1.mean(axis=None)
    assert_eq(result, expected, check_dtype=not DASK_EXPR_ENABLED)
    if not DASK_EXPR_ENABLED:
        assert_dask_graph(ddf1.sum(split_every=split_every), 'dataframe-sum')
        assert_dask_graph(ddf1.prod(split_every=split_every), 'dataframe-prod')
        assert_dask_graph(ddf1.min(split_every=split_every), 'dataframe-min')
        assert_dask_graph(ddf1.max(split_every=split_every), 'dataframe-max')
        assert_dask_graph(ddf1.count(split_every=split_every), 'dataframe-count')
        assert_dask_graph(ddf1.std(split_every=split_every), 'dataframe-var')
        assert_dask_graph(ddf1.std(split_every=split_every), 'moment_chunk')
        assert_dask_graph(ddf1.std(split_every=split_every), 'moment_agg')
        assert_dask_graph(ddf1.std(split_every=split_every), 'values')
        assert_dask_graph(ddf1.var(split_every=split_every), 'moment_chunk')
        assert_dask_graph(ddf1.var(split_every=split_every), 'moment_agg')
        assert_dask_graph(ddf1.var(split_every=split_every), 'values')
        assert_dask_graph(ddf1.sem(split_every=split_every), 'dataframe-var')
        assert_dask_graph(ddf1.sem(split_every=split_every), 'moment_chunk')
        assert_dask_graph(ddf1.sem(split_every=split_every), 'moment_agg')
        assert_dask_graph(ddf1.sem(split_every=split_every), 'values')
        assert_dask_graph(ddf1.mean(split_every=split_every), 'dataframe-sum')
        assert_dask_graph(ddf1.mean(split_every=split_every), 'dataframe-count')
        assert_dask_graph(ddf1.sum(axis=1, split_every=split_every), 'dataframe-sum')
        assert_dask_graph(ddf1.prod(axis=1, split_every=split_every), 'dataframe-prod')
        assert_dask_graph(ddf1.min(axis=1, split_every=split_every), 'dataframe-min')
        assert_dask_graph(ddf1.max(axis=1, split_every=split_every), 'dataframe-max')
        assert_dask_graph(ddf1.count(axis=1, split_every=split_every), 'dataframe-count')
        assert_dask_graph(ddf1.std(axis=1, split_every=split_every), 'dataframe-std')
        assert_dask_graph(ddf1.var(axis=1, split_every=split_every), 'dataframe-var')
        assert_dask_graph(ddf1.sem(axis=1, split_every=split_every), 'dataframe-sem')
        assert_dask_graph(ddf1.mean(axis=1, split_every=split_every), 'dataframe-mean')