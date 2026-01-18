from __future__ import annotations
from itertools import product
import pandas as pd
import pytest
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_220
from dask.dataframe.utils import assert_eq
def test_resample_throws_error_when_parition_index_does_not_match_index():
    index = pd.date_range('1-1-2000', '2-15-2000', freq='D')
    index = index.union(pd.date_range('4-15-2000', '5-15-2000', freq='D'))
    ps = pd.Series(range(len(index)), index=index)
    ds = dd.from_pandas(ps, npartitions=5)
    with pytest.raises(ValueError, match='Index is not contained within new index.'):
        ds.resample(f'2{ME}').count().compute()