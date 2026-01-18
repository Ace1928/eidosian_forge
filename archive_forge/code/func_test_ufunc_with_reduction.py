from __future__ import annotations
import warnings
import pytest
import numpy as np
import dask.array as da
import dask.dataframe as dd
from dask.dataframe.utils import assert_eq
@pytest.mark.parametrize('redfunc', ['sum', 'prod', 'min', 'max', 'mean'])
@pytest.mark.parametrize('ufunc', _BASE_UFUNCS)
@pytest.mark.parametrize('pandas', [pd.Series(np.abs(np.random.randn(100))), pd.DataFrame({'A': np.random.randint(1, 100, size=20), 'B': np.random.randint(1, 100, size=20), 'C': np.abs(np.random.randn(20))})])
def test_ufunc_with_reduction(redfunc, ufunc, pandas):
    dask = dd.from_pandas(pandas, 3)
    np_redfunc = getattr(np, redfunc)
    np_ufunc = getattr(np, ufunc)
    if redfunc == 'prod' and ufunc in ['conj', 'square', 'negative', 'absolute'] and isinstance(pandas, pd.DataFrame):
        pytest.xfail("'prod' overflowing with integer columns in pandas 1.2.0")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        warnings.simplefilter('ignore', FutureWarning)
        if dd._dask_expr_enabled():
            import dask_expr as dx
            assert isinstance(np_redfunc(dask), (dd.DataFrame, dd.Series, dx.Scalar))
        else:
            assert isinstance(np_redfunc(dask), (dd.DataFrame, dd.Series, dd.core.Scalar))
        assert_eq(np_redfunc(np_ufunc(dask)), np_redfunc(np_ufunc(pandas)))