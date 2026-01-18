from __future__ import annotations
import pandas as pd
import pytest
import dask
import dask.dataframe as dd
from dask.blockwise import Blockwise, optimize_blockwise
from dask.dataframe._compat import PANDAS_GE_200, PANDAS_GE_220, tm
from dask.dataframe.optimize import optimize_dataframe_getitem
from dask.dataframe.utils import assert_eq, get_string_dtype
def test_with_spec_pyarrow():
    pytest.importorskip('pyarrow', '1.0.0', reason='pyarrow is required')
    from dask.dataframe.io.demo import ColumnSpec, DatasetSpec, with_spec
    spec = DatasetSpec(npartitions=1, nrecords=10, column_specs=[ColumnSpec(dtype='string[pyarrow]', length=10, random=True)])
    ddf = with_spec(spec, seed=42)
    assert isinstance(ddf, dd.DataFrame)
    assert ddf.columns.tolist() == ['string_pyarrow1']
    assert ddf['string_pyarrow1'].dtype == 'string[pyarrow]'
    res = ddf.compute()
    assert res['string_pyarrow1'].dtype == 'string[pyarrow]'
    assert all((len(s) == 10 for s in res['string_pyarrow1'].tolist()))