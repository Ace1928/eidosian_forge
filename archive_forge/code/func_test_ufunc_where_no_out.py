from __future__ import annotations
import pickle
import warnings
from functools import partial
from operator import add
import pytest
import dask.array as da
from dask.array.ufunc import da_frompyfunc
from dask.array.utils import assert_eq
from dask.base import tokenize
def test_ufunc_where_no_out():
    left = np.arange(4)
    right = np.arange(4, 8)
    where = np.array([[0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 0, 1]]).astype('bool')
    d_where = da.from_array(where, chunks=2)
    d_left = da.from_array(left, chunks=2)
    d_right = da.from_array(right, chunks=2)
    expected = np.add(left, right, where=where)
    result = da.add(d_left, d_right, where=d_where)
    expected_masked = np.where(where, expected, 0)
    result_masked = np.where(where, expected, 0)
    assert_eq(expected_masked, result_masked)
    expected_no_where = np.add(left, right)
    assert not np.equal(result.compute(), expected_no_where).all()