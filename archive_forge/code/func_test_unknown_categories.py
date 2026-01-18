from __future__ import annotations
import contextlib
import operator
import warnings
import numpy as np
import pandas as pd
import pytest
import dask
import dask.dataframe as dd
from dask.dataframe import _compat
from dask.dataframe._compat import (
from dask.dataframe._pyarrow import to_pyarrow_string
from dask.dataframe.core import _concat
from dask.dataframe.utils import (
@pytest.mark.parametrize('series', cat_series)
def test_unknown_categories(self, series):
    a, da = series
    assert da.cat.known
    da = da.cat.as_unknown()
    assert not da.cat.known
    with pytest.raises(NotImplementedError, match='with unknown categories'):
        da.cat.categories
    with pytest.raises(NotImplementedError, match='with unknown categories'):
        da.cat.codes
    with pytest.raises(AttributeError, match='with unknown categories'):
        da.cat.categories
    with pytest.raises(AttributeError, match='with unknown categories'):
        da.cat.codes
    db = da.cat.set_categories(['a', 'b', 'c'])
    assert db.cat.known
    tm.assert_index_equal(db.cat.categories, get_cat(a).categories)
    assert_array_index_eq(db.cat.codes, get_cat(a).codes)
    db = da.cat.as_known()
    assert db.cat.known
    res = db.compute()
    tm.assert_index_equal(db.cat.categories, get_cat(res).categories)
    assert_array_index_eq(db.cat.codes, get_cat(res).codes)