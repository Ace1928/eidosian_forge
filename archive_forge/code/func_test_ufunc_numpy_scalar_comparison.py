from __future__ import annotations
import warnings
import pytest
import numpy as np
import dask.array as da
import dask.dataframe as dd
from dask.dataframe.utils import assert_eq
@pytest.mark.parametrize('pandas', [pd.Series(np.random.randint(1, 100, size=100)), pd.DataFrame({'A': np.random.randint(1, 100, size=20), 'B': np.random.randint(1, 100, size=20), 'C': np.abs(np.random.randn(20))})])
@pytest.mark.parametrize('scalar', [15, 16.4, np.int64(15), np.float64(16.4)])
def test_ufunc_numpy_scalar_comparison(pandas, scalar):
    dask_compare = scalar >= dd.from_pandas(pandas, npartitions=3)
    pandas_compare = scalar >= pandas
    assert_eq(dask_compare, pandas_compare)