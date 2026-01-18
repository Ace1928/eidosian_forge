from __future__ import annotations
import warnings
import pytest
import numpy as np
import dask.array as da
import dask.dataframe as dd
from dask.dataframe.utils import assert_eq
@pytest.mark.parametrize('ufunc', _UFUNCS_2ARG)
@pytest.mark.parametrize('make_pandas_input', [lambda: pd.Series(np.random.randint(1, 100, size=20)), lambda: pd.DataFrame(np.random.randint(1, 100, size=(20, 2)), columns=['A', 'B'])])
def test_ufunc_with_2args(ufunc, make_pandas_input):
    dafunc = getattr(da, ufunc)
    npfunc = getattr(np, ufunc)
    pandas1 = make_pandas_input()
    pandas2 = make_pandas_input()
    dask1 = dd.from_pandas(pandas1, 3)
    dask2 = dd.from_pandas(pandas2, 4)
    pandas_type = pandas1.__class__
    dask_type = dask1.__class__
    assert isinstance(dafunc(dask1, dask2), dask_type)
    assert_eq(dafunc(dask1, dask2), npfunc(pandas1, pandas2))
    assert isinstance(dafunc(dask1, pandas2), dask_type)
    assert_eq(dafunc(dask1, pandas2), npfunc(pandas1, pandas2))
    if isinstance(npfunc, np.ufunc):
        assert isinstance(npfunc(dask1, dask2), dask_type)
        assert isinstance(npfunc(dask1, pandas2), dask_type)
    else:
        assert isinstance(npfunc(dask1, dask2), pandas_type)
        assert isinstance(npfunc(dask1, pandas2), pandas_type)
    assert_eq(npfunc(dask1, dask2), npfunc(pandas1, pandas2))
    assert_eq(npfunc(dask1, pandas2), npfunc(pandas1, pandas2))
    assert isinstance(dafunc(pandas1, pandas2), pandas_type)
    assert_eq(dafunc(pandas1, pandas2), npfunc(pandas1, pandas2))